import os
import pathlib
import itertools
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc_utils

# gc prefix added to avoid name conflict with other modules
# This file contains all IO related functions



def read_tecan_stacker_xlsx(file_path, plate_rows, plate_columns, log):
    '''
    Desrciption
    -----------
    Read the content of a xlsx file in a tecan stakcer format
    
    Parameters
    ----------
    file_path : str
        The path to the file
    plate_rows : list of strings
        The rows of the plates to read data from
    plate_columns : list of integers
        The columns of the plates to read data from
    log : list
        The list to which the log messages will be appended
    Returns
    -------
    pandas.DataFrame
        dataframe containing the data from the input file indxed by file_name, plate_name, row_index, column_index. Columns:
        - ``file_name`` (:py:class:`str`): the name of the file the data comes from.
        - ``plate_name`` (:py:class:`str`): the name of the plate the data comes from.
        - ``well_key`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``well_row_index`` (:py:class:`int`): the well row index.
        - ``well_column_index`` (:py:class:`str`): the well column index.
        - ``time`` (:py:class:`float`) measurement time in hours
        - ``temperature`` (:py:class:`float`) degrees in celsius
        - ``OD`` (:py:class:`float`) Optical density in AU
    '''

    current_file_name = pathlib.Path(file_path).stem
    print(f"Reading data from file {current_file_name}")

    file_names = []
    plate_names = []
    well_keys = []
    well_column_indexes = []
    well_row_indexes = []
    times = []
    temperatures = []
    ODs = []
    

    with pd.ExcelFile(file_path) as excel_file:
        # Loop all the sheets in the file
        for sheet_name in excel_file.sheet_names:
            # Hold all the first OD values with the key being the well name
            initial_ODs = {}

            # Load the current sheet of the excel file
            raw_data_df = pd.read_excel(excel_file, sheet_name)
                        
            # Shared variables each measurement cycle
            cycle_time_in_hours = 0
            cycle_temp = 0

            # Loop all the rows in the dataframe
            for row in raw_data_df.itertuples(index=False):
                # Save the time of the measurement in hours
                if row[0] == "Time [s]":
                    cycle_time_in_hours = (row[1] / 3600)
                # Save the temperature at the time of measurement
                elif row[0] == "Temp. [°C]":
                    cycle_temp = row[1]
                # Save the OD values
                elif row[0] in plate_rows:
                    row_letter = row[0]

                    for current_column in range(0, raw_data_df.shape[1]):
                        if current_column in plate_columns:
                            # if OD is too high it will cause an error when measuring therefore raise an error
                            if row[current_column] == "OVER":
                                err_msg = f'A measurement with the value of "OVER" is in cell {str(((row[0])))}{str(current_column)} at sheet: {sheet_name} in file: {current_file_name}'
                                log.append(err_msg)
                                raise ValueError(err_msg)
                            
                            # Check if a value has alredy been saved for the current well
                            well_loc = f'{row_letter}{current_column}'
                            if not well_loc in initial_ODs:
                                # Save the initial OD value for the current well to later normalize against
                                initial_ODs[well_loc] = row[current_column]
                                
                            # Add data to the lists
                            file_names.append(current_file_name)
                            plate_names.append(sheet_name)
                            # The current_column is given in 1 based indexing, convert to 0 based indexing for consistency
                            well_column_indexes.append(current_column - 1)
                            well_row_indexes.append(gc_utils.convert_row_letter_to_number(row_letter))
                            well_keys.append(well_loc)
                            times.append(cycle_time_in_hours)
                            temperatures.append(cycle_temp)

                            # Normalize the OD by subtracting the initial OD (for the first OD read the the value will be negative therefore set it to 0)
                            ODs.append(row[current_column] - initial_ODs[well_loc] if row[current_column] - initial_ODs[well_loc] > 0 else 0)

    raw_data_df = pd.DataFrame({
        "file_name": file_names, "plate_name": plate_names, "well_key": well_keys , "well_row_index": well_row_indexes,
        "well_column_index": well_column_indexes, "time": times, "temperature": temperatures, "OD": ODs
    })

    raw_data_df = raw_data_df.set_index(["file_name", "plate_name", "well_row_index", "well_column_index"])
    # Sort the index
    return raw_data_df.sort_index()

def save_dataframe_to_csv(df, output_file_path, file_name):
    '''
    Description
    -----------
    Save a dataframe to a csv file with the indexes
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be saved
    output_file_path : str
        The path to the folder where the csv file will be saved
    file_name : str
        The name of the csv file, supply the value of the file name without the extension
    '''
    #Create the output file path with the file name and extension
    file_path_with_file_name = os.path.join(output_file_path, f'{file_name}.csv')
    # Save the dataframe a csv file
    df.to_csv(file_path_with_file_name, index=True)
    return file_path_with_file_name


def create_directory(father_directory, nested_directory_name):
    '''
    Description
    -----------
    Create a directory if it does not exist
    
    Parameters
    ----------
    father_directory : str
        The path to the directory under which the new directory will be created
    nested_directory_name : str
        The name of the nested directory to be created
    '''
    # Create the output directory path
    new_dir_path = os.path.join(father_directory, nested_directory_name)
    # Create the directory if it does not exist
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path


def create_single_well_graphs(file_name ,raw_data, summary_data, output_path, title, decimal_percision):
    '''Create graphs from the data collected in previous steps for each well in the experiment
    Parameters
    ----------
    file_name : str
        The name of the file being processed. Will be used to prefix the output file names
    raw_data : pandas.DataFrame
        dataframe returned from the read_tecan_stacker_xlsx function or one with the same structure
    summary_data : pandas.DataFrame
        dataframe returned from the get_experiment_growth_parameters function or one with the same structure
    output_path : str
        Save path
    title: str
        The title for the graphs
    decimal_percision: int
        The amount of digits after the decimal point to show in the labels
    Returns
    -------
    null
    '''

    # Matplotlib backend mode - a non-interactive backend that can only write to files
    # Before changing to this mode the program would crash after the creation of about 250 graphs
    matplotlib.use("Agg")

    plt.style.use('ggplot')

    # Styles
    point_size = 50
    alpha = 0.6

    df_unindexed = raw_data.reset_index()
    # Get all unique keys to loop through
    file_names = df_unindexed['file_name'].unique()
    plate_names = df_unindexed['plate_name'].unique()
    well_row_indexes = df_unindexed['well_row_index'].unique()
    well_column_indexes = df_unindexed['well_column_index'].unique()


    
    for file_name, plate_name, well_row_index, well_column_index in itertools.product(file_names ,plate_names, well_row_indexes, well_column_indexes):
        well_raw_data = raw_data.xs((file_name, plate_name, well_row_index, well_column_index), level=['file_name', 'plate_name', 'well_row_index', 'well_column_index'])
        well_summary_data = (summary_data.xs((file_name, plate_name, well_row_index, well_column_index), level=['file_name', 'plate_name', 'well_row_index', 'well_column_index'])).iloc[0,:]
        

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('OD600')

        ax.plot(well_raw_data["time"], well_raw_data["OD"], color='black')
        
        # If the well is valid graph it with the data from the fitting procedure, otherwise only graph time vs OD as an aid for seeing what went wrong
        if well_summary_data["is_valid"]:
            lag_end_time, lag_end_OD = well_summary_data["lag_end_time"], well_summary_data["lag_end_OD"]
            # lag data
            ax.scatter([lag_end_time], [lag_end_OD], s=point_size ,alpha=alpha, c=['purple'], marker='s',
                        label= f'end of leg phase: {str(round(lag_end_time, decimal_percision))} hours')

            # Max population growth rate
            max_population_gr_time, max_population_gr_OD, max_population_gr_slope = well_summary_data["max_population_gr_time"], well_summary_data["max_population_gr_OD"], well_summary_data["max_population_gr_slope"]
            # Plot the point and the linear function matching the max population growth rate
            ax.axline((max_population_gr_time, max_population_gr_OD), slope=max_population_gr_slope, color='blue', linestyle=':')
            # plot the point on the graph at which this occures
            ax.scatter([max_population_gr_time], [max_population_gr_OD], c=['darkblue'], s=point_size, alpha=alpha, label=f'Min doubling time: {round(well_summary_data["min_doubling_time"], decimal_percision)} hours')

            # End of exponential phase
            exponet_end_time, exponet_end_OD = well_summary_data["exponet_end_time"], well_summary_data["exponet_end_OD"]
            # plot the point with the label
            ax.scatter([exponet_end_time], [exponet_end_OD], c=["brown"], marker='d' ,s=point_size ,alpha=alpha, label=f'95% of growth: {str(round(exponet_end_time, decimal_percision))} hours')

            carrying_capacity = well_summary_data["carrying_capacity"]
            ax.axhline(y=carrying_capacity, color='black', linestyle='dashdot', label=f'Carrying capacity: {(round(carrying_capacity, decimal_percision))}')

            ax.legend(loc="lower right")
        
        # Save the figure
        fig.savefig(os.path.join(output_path, f"well {well_summary_data['well_key']} from {plate_name} in {file_name}.png"))
        plt.close("all")


def import_previous_run_data(output_path):
    '''
    Desrciption
    -----------
    Read the content the output file with the results of a previous run.
    The folder must include the multiple repeat comprison df with it's originial name,
    At least one raw data df (as saved from the read_tecan_stacker_xlsx or any future version of the function that outputs the same file structure)
    and the same number of summary files belonging to the same original file. If any of these assumptions is not true the function will raise a value error.
    
    Parameters
    ----------
    output_path : str
        The path to the output file with the results of a previous run
    
    Returns
    -------
    file_raw_data_df_mapping : dictionary
        The name of the file as the key and the raw data from the file as a pandas.DataFrame as the value
    file_summary_df_mapping : dictionary
        The name of the file as the key and the summary data from get_experiment_growth_parameters as a pandas.DataFrame as the value
    variation_matrix : pandas.DataFrame
        The muliple repeat comprison table as returned from get_reps_variation_data
    '''
    
    # Initialize the dictionaries for raw data and summary data
    file_raw_data_df_mapping = {}
    file_summary_df_mapping = {}
    variation_matrix = None
    
    # List all files in the specified directory
    all_files = os.listdir(output_path)
    
    # Filter files based on the required suffix
    raw_data_files = [f for f in all_files if f.endswith('_raw_data.csv')]
    summary_data_files = [f for f in all_files if f.endswith('_summary_data.csv')]
    variation_data_files = [f for f in all_files if f.endswith('_coupled_reps_data.csv')]
    
    # Check that the number of raw data files matches the number of summary data files
    if len(raw_data_files) != len(summary_data_files):
        raise ValueError("The number of raw data files does not match the number of summary data files.")
    
    # Check that there is exactly one variation matrix file
    if len(variation_data_files) != 1:
        raise ValueError("There must be exactly one '_coupled_reps_data.csv' file.")
    
    # Process raw data files
    for raw_file in raw_data_files:
        file_base_name = raw_file.replace('_raw_data.csv', '')
        raw_data_path = os.path.join(output_path, raw_file)
        file_raw_data_df_mapping[file_base_name] = pd.read_csv(raw_data_path)
    
    # Process summary data files
    for summary_file in summary_data_files:
        file_base_name = summary_file.replace('_summary_data.csv', '')
        summary_data_path = os.path.join(output_path, summary_file)
        file_summary_df_mapping[file_base_name] = pd.read_csv(summary_data_path)
    
    # Load the variation matrix
    variation_matrix_path = os.path.join(output_path, variation_data_files[0])
    variation_matrix = pd.read_csv(variation_matrix_path)
    
    
    return file_raw_data_df_mapping, file_summary_df_mapping, variation_matrix


def create_reps_avarage_graphs(reps, averaged_reps, output_path):
    return 1