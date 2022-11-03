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



def read_tecan_stacker_xlsx(file_path, data_rows=["A" ,"B", "C", "D" ,"E", "F", "G", "H"], data_columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    '''
    Desrciption
    -----------
    Read the content of a xlsx file in a tecan stakcer format
    
    Parameters
    ----------
    file_path : str
        The path to the folder where all the data for analysis is stored
    data_rows : list of strings
        The rows of the xlsx file that are to be read
    data_columns : list of integers
        The columns of the xlsx file that are to be read
    Returns
    -------
    pandas.DataFrame
        The dataframe containing the data from the xlsx file. Data frame containing the columns:
        - ``file_name`` (:py:class:`str`): the name of the file that is being analysed.
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``Time`` (:py:class:`float`, in hours)
        - ``OD`` (:py:class:`float`, in AU)
        - ``temperature`` (:py:class:`float` in celcius)
    '''
    # Get logger object
    logger = gc_utils.get_logger()

    # data will hold all the measurements form the current file
    data = None
    # Hold the index to hold the next measurement into the numpy array
    curr_index = 0
    # calculate what the size of data needs to be to initialize it once
    with pd.ExcelFile(file_path) as excel_file:
        df = pd.read_excel(excel_file, excel_file.sheet_names[0])
        number_of_measurements = gc_utils.get_max_cycle_number(df)
        data = np.empty((len(data_rows) * len(data_columns) * number_of_measurements * len(excel_file.sheet_names), 6), dtype=object)
    current_file_name = pathlib.Path(file_path).stem
    print(f"Reading data from file {current_file_name}")
    
    with pd.ExcelFile(file_path) as excel_file:
        # Loop all the sheets in the file
        for sheet_name in excel_file.sheet_names:
            # Hold all the first OD values with the key being the well name
            initial_ODs = {}

            # Load the current sheet of the excel file
            df = pd.read_excel(excel_file, sheet_name)
                        
            # Shared variables each measurement cycle
            cycle_time_in_hours = 0
            cycle_temp = 0

            # Loop all the rows in the dataframe
            for row in df.itertuples(index=False):
                # Save the time of the measurement in hours
                if row[0] == "Time [s]":
                    cycle_time_in_hours = (row[1] / 3600)
                # Save the temperature at the time of measurement
                elif row[0] == "Temp. [°C]":
                    cycle_temp = row[1]
                
                # Save the OD of the wells
                # if the current row is in the list of rows to be read
                elif row[0] in data_rows:
                    # Save the letter that represents the row
                    row_letter = row[0]

                    # column_number is used to track the current column beigng processed
                    for current_column in range (0, df.shape[1]):
                        # Only read the data in the columns that are in the list of columns to be read
                        if current_column in data_columns:
                            # if OD is too high it will cause an error when measuring therefore raise an error
                            if row[current_column] == "OVER":
                                err_msg = f'A measurement with the value of "OVER" is in cell {str(((row[0])))}{str(current_column)} at sheet: {sheet_name} in file: {current_file_name}'
                                + ' please correct the invalid value and try rerunning the analysis'
                                logger.critical(err_msg)
                                raise ValueError(err_msg)
                            else:
                                # Check if a value has alredy been saved for the current well
                                key = f'{row_letter}{current_column}'
                                if not key in initial_ODs:
                                    # Save the initial OD value for the current well to later normalize against
                                    initial_ODs[key] = row[current_column]
                                    # Save the OD of the current well and all other data attached to it in a new index in the data array
                                    # Since the first measurement is considered the "zero" is this experiment then set the OD value as 0
                                    # Row order: file_name, plate, well, time, OD, temperature
                                    data[curr_index] = [current_file_name, sheet_name, key, cycle_time_in_hours, 0, cycle_temp]
                                    # Increase the index tracking the current insertion into the numpy array
                                    curr_index += 1
                                else:
                                    # Save the OD of the current well and all other data attached to it in a new index in the data array and noramalize it
                                    # Row order: file_name, plate, well, time, OD, temperature
                                    data[curr_index] = [current_file_name, sheet_name, key, cycle_time_in_hours, row[current_column] - initial_ODs[key], cycle_temp]
                                    # Increase the index tracking the current insertion into the numpy array
                                    curr_index += 1
    # Create a dataframe from the numpy array and returrn it
    df = pd.DataFrame(data, columns=["file_name", "plate", "well", "Time", "OD", "temperature"])
    df = df.astype({ 'file_name': str, 'plate': str, 'well': str, 'Time': float, 'OD': float, 'temperature': float })
    # Make sure there are no empty rows in the dataframe
    df = df.dropna()
    # Index the dataframe by the file_name, plate and well and return
    df.set_index(["file_name", "plate", "well"], inplace=True)
    # Sort the index
    return df.sort_index()

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

    # Styles
    point_size = 50
    alpha = 0.6

    df_unindexed = raw_data.reset_index()
    # Get all the file_names, platenames and wellnames from the dataframe to iterate over
    # file_name + platename + wellname are the keys by which the df is indexed
    file_names = df_unindexed['file_name'].unique()
    plate_names = df_unindexed['plate'].unique()
    well_names = df_unindexed['well'].unique()

    
    for file_name ,plate_name, well_name in itertools.product(file_names ,plate_names, well_names):
        well_raw_data = raw_data.xs((file_name, plate_name, well_name), level=['file_name', 'plate', 'well'])
        well_summary_data = (summary_data.xs((file_name, plate_name, well_name), level=['file_name', 'plate', 'well'])).iloc[0,:]
        # Setup axis and the figure objects
        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('OD600')

        # Plot the main graph
        ax.plot(well_raw_data["Time"], well_raw_data["OD"])
        
        # If the well is valid graph it with the data from the fitting procedure, otherwise only graph time vs OD as an aid for seeing what went wrong
        if well_summary_data["is_valid"]:
            lag_end_time, lag_end_OD = well_summary_data["lag_end_time"], well_summary_data["lag_end_OD"]
            # plot the point with the label
            ax.scatter([lag_end_time], [lag_end_OD], s=point_size ,alpha=alpha, 
                        label= f'end of leg phase: {str(round(lag_end_time, decimal_percision))} hours')

            # Max population growth rate plotting
            max_population_gr_time, max_population_gr_OD, max_population_gr_slope = well_summary_data["max_population_gr_time"], well_summary_data["max_population_gr_OD"], well_summary_data["max_population_gr_slope"]
            # Plot the point and the linear function matching the max population growth rate
            ax.axline((max_population_gr_time, max_population_gr_OD), slope=max_population_gr_slope, color='firebrick', linestyle=':', label=f'maximum population growth rate: {(round(max_population_gr_slope, decimal_percision))}')
            # plot the point on the graph at which this occures
            ax.scatter([max_population_gr_time], [max_population_gr_OD], c=['firebrick'], s=point_size, alpha=alpha)

            # End of exponential phase
            exponet_end_time, exponet_end_OD = well_summary_data["exponet_end_time"], well_summary_data["exponet_end_OD"]
            # plot the point with the label
            ax.scatter([exponet_end_time], [exponet_end_OD], c=["darkgreen"], s=point_size ,alpha=alpha, label=f'95% of growth: {str(round(exponet_end_time, decimal_percision))} hours')

            carrying_capacity = well_summary_data["carrying_capacity"]
            ax.axhline(y=carrying_capacity, color='black', linestyle=':', label=f'Carrying capacity: {(round(carrying_capacity, decimal_percision))}')

            ax.legend(loc="lower right")
        
        # Save the figure
        fig.savefig(os.path.join(output_path, f"well {well_name} from {plate_name} in {file_name}"))
        plt.close("all")