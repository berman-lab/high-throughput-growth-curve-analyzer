import os
import logging
import pathlib
import curveball
import itertools
import matplotlib
import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt

# Interanal modules
import gc_io
import gc_utils

# To be deleted
from well_data import WellData
from experiment_data import ExperimentData

def main():
    # Base config and program parametrs
    base_path = os.path.normcase("c:\Data\\bio-graphs")
    # Input directory
    input_directory = os.path.join(base_path, "In")
    # The directory into which all the graphs will be saved
    output_directory = os.path.join(base_path, "Out")
    
    # Setup log file
    logging.basicConfig(filename=f'{os.path.join(output_directory, "messages.log")}', filemode='w', encoding='utf-8', level=logging.DEBUG)

    # Globals
    # Valued used to replace zeros with a values close to 0
    # that will not cause errors when applaying log to the data
    global ZERO_SUB
    ZERO_SUB = 0.000001

    # Used to account for the extrs column that the pandas reading function adds to the data
    global LEFT_OFFSET
    LEFT_OFFSET = 1

    # The amount of digits after the decimal point to show in plots
    global DECIMAL_PERCISION_IN_PLOTS
    DECIMAL_PERCISION_IN_PLOTS = 3

    # Matplotlib backend mode - a non-interactive backend that can only write to files
    # Before changing to this mode the program would crash after the creation of about 250 graphs
    matplotlib.use("Agg")    

    # Get all the files from the input directory
    files_for_analysis = gc_utils.get_files_from_directory(input_directory, "xlsx")
    # Crearte a dictionary of all the files with the file name as the key and all the measurements in a data as the value in a dataframe
    file_df_mapping = {}

    print("Importing the data from files")
    # Add all the file names as keys to the dictionary and save the data in the dictionary
    for file in files_for_analysis:
        file_df_mapping[pathlib.Path(file).stem] = gc_io.read_tecan_stacker_xlsx(file, data_rows=["B", "C", "D" ,"E", "F", "G"], data_columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11] )

    # Save the raw data to a xlsx file

    # Analysis of the data
    fill_growth_parameters(raw_data)

    print("Creating figures")
    # Graph the data and save the figures to the output_directory
    create_single_well_graphs(raw_data, output_directory, "OD600[nm] against time[sec]")

    df_raw_data, df_wells_summary = create_data_tables(raw_data, output_directory)

    df_raw_data.to_csv(os.path.join(output_directory, f'{raw_data[0].file_name}_raw_data.csv'), index=False, encoding='utf-8')
    df_wells_summary.to_csv(os.path.join(output_directory, f'{raw_data[0].file_name}_summary.csv'), index=False, encoding='utf-8')

    raw_data = get_csv_raw_data(input_directory)
    variation_matrix = get_reps_variation_data(raw_data)
    variation_matrix.to_csv(os.path.join(output_directory, f'{raw_data[0][0].file_name}_coupled_reps_data.csv'), index=False, encoding='utf-8')
    averaged_rep = get_averaged_ExperimentData(raw_data)
    create_reps_avarage_graphs(raw_data, averaged_rep, output_directory)

    save_err_log(output_directory, "Error log")

#IO
def get_tecan_stacker_data(input_directory, extensions, err_log, data_rows=["B", "C", "D" ,"E", "F", "G"], data_columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    '''
    Desrciption
    -----------
    Read all the data from the files with the given extension in the input directory given
    
    Parameters
    ----------
    input_directory : path object
        The path to the folder where all the data we want to analyze is stored
    extensions : (str, str, ...)
        tuple with all the files with a given file extension we wish to include in the analysis
    err_log : [str]
        a refernce to the list containing all the previosuly logged errors
    data_rows:
        A list of all the names of the rows to be analysed, defaults to A to H as in a normal 96 well plate
  

    Returns
    -------
    ExperimentData object
    '''

    # The container of the data
    parsed_data = []

    # retrive all the data files by extesions from the In directory
    excel_files_paths = get_files_from_directory(input_directory, extensions)

    # Loop excel_files_locations list to read all the relevant files
    for excel_file_location in excel_files_paths:
        # Take the excel_file_location and use it to initiate an ExcelFile object as the context
        with pd.ExcelFile(excel_file_location) as excel_file:
            # Loop all the sheets in the file
            for sheet in excel_file.sheet_names:
                try:
                    # Get the name of the current file. The last part of the path then remove the file extension
                    curr_file_name = pathlib.Path(excel_file_location).stem
                    
                    # Create a new object to save data into
                    parsed_data.append(ExperimentData(plate_name=sheet, file_name=curr_file_name))

                    # Load the current sheet of the excel file
                    df = pd.read_excel(excel_file, sheet)
                    

                    # run tourgh all the rows and columns and save the data into object for graphing later
                    # We use 96 well plates but only use the inner wells. That is, we treat the 96 well as a 60 well (6 X 10)
                    for row in df.itertuples():
                        # save the time of reading from the start of the experiment in seconds
                        if row[1] == "Time [s]":
                            parsed_data[-1].times.append(row[2] / 3600)
                        # save the temperature at the time of reading
                        elif row[1] == "Temp. [°C]":
                            parsed_data[-1].temps.append(row[2])
                        # save the OD of the well
                        
                        elif row[1] in data_rows:
                            # Convert the character index to numaric index to be used to insert under the desired key in ODs
                            # 66 is the ASCII value of B and afterwards all the values are sequencial
                            row_index = ord(row[1]) - 66

                            # This offset comes from the fact that in this expiremnt we don't right-most column and the index given by itertuples
                            LEFT_OFFSET = 1
                            # Push all the values in the list with the needed change set by the LEFT_OFFSET
                            data_columns_offset = [column_index + LEFT_OFFSET for column_index in data_columns]
                            # Collect all values from the columns to ODs
                            for i in data_columns_offset:
                                # i is the index of the relevant cell within the excel sheet j is the adjusted value to make it zero based index to be used when saving to ODs
                                j = i - LEFT_OFFSET
                                curr_well = (row_index, j)
                                if curr_well not in parsed_data[-1].wells:
                                    parsed_data[-1].wells[curr_well] = WellData(is_valid=False, ODs=[row[i]])
                                # There is a previous reading for this cell, therefore normalize it against the first read then save it
                                else:
                                    if row[i] == "OVER":
                                        raise ValueError(f'a measurement with the value of OVER is in cell {str(((row[1]), j + LEFT_OFFSET))} at sheet: {sheet} please fix and try again')
                                    else:
                                        parsed_data[-1].wells[curr_well].ODs.append(row[i] - parsed_data[-1].wells[curr_well].ODs[0])
                except Exception as e:
                    print(str(e))
                    #f"data read at sheet {sheet} at file {curr_file_name} failed with the following exception mesaage: {str(e)}")
    # Zero out all the first ODs since normalization was done in relation to them and it's finished, therefore they need to be set to 0
    for experiment_data in parsed_data:
        for row_index, column_index in experiment_data.wells:
            experiment_data.wells[(row_index, column_index)].ODs[0] = 0
    
    return parsed_data

def get_csv_raw_data(input_directory, extensions, err_log, data_rows=["B", "C", "D" ,"E", "F", "G"], data_columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    '''Read the data from the previously exported csv with the given extension in the input directory given
    
     Parameters
    ----------
    input_directory : path object
        The path to the folder where all the data we want to analyze is stored
    extensions : (str, str, ...)
        tuple with all the files with a given file extension we wish to include in the analysis
    err_log : [str]
        a refernce to the list containing all the previosuly logged errors
    data_rows:
        A list of all the names of the rows to be analysed, defaults to A to H as in a normal 96 well plate
  

    Returns
    -------
    ExperimentData object
    '''
    # list of the data as expremint_data objects
    parsed_data = []
    # Save all the data from a replicate into
    rep_data = []
    
    csvs_input_paths = get_files_from_directory(input_directory, extensions)
    # Get all csv files with 'raw_data' in their name
    csv_raw_data_paths = list(filter(lambda file_name: 'raw_data' in file_name, csvs_input_paths))
    csv_summary_paths = list(filter(lambda file_name: 'summary' in file_name, csvs_input_paths))

    # Load all summary csvs into one data frame
    summary_csvs = []
    for csv_summary_path in csv_summary_paths:
        summary_csvs.append(pd.read_csv(csv_summary_path))

    summary_csvs = pd.concat(summary_csvs, ignore_index=True, sort=False)   
    summary_csvs["plate"] = summary_csvs["plate"].str.lower()
    # Save all plate names
    plate_names = pd.unique(summary_csvs['plate'])
    file_names = pd.unique(summary_csvs['filename'])

    # get the amount of plates in the first experiment
    plate_count = len(pd.unique(summary_csvs.loc[summary_csvs['filename'] == summary_csvs['filename'][0]]['plate']))

    #Check if there are too many plates, i.e not the same names between replicates
    if len(plate_names) > plate_count:
        add_line_to_error_log(err_log, 'Too many plates in summary_csvs, please make sure all experiment excel files have the same sheet names in them')
        raise ValueError(f'Too many plates in summary_csvs, please make sure all experiment excel files have the same sheet names in them')

    # Check if there are too many filenames inside the dataframe
    if len(file_names) != len(csv_summary_paths):
        add_line_to_error_log(err_log, 'Too many file names in summary_csvs')
        raise ValueError(f'Too many file names in summary_csvs')

    # go through the csvs and move the data into parsed_data
    for i, raw_csv_file_location in enumerate(csv_raw_data_paths):
        rep_data = []

        current_csv = pd.read_csv(raw_csv_file_location)
        current_csv['plate'] = current_csv['plate'].str.lower()

        for plate in plate_names:
            curr_plate_raw_data = current_csv.loc[current_csv['plate'] == plate]

            # All wells in the same plate have the same times and temperatures, therefore get the data with the first well the user specified
            curr_plate_first_well_df = curr_plate_raw_data.loc[curr_plate_raw_data['well'] == f'{data_rows[0]}{data_columns[0]}']
            plate_measurements_times = list(curr_plate_first_well_df['time'])
            plate_temps = list(curr_plate_first_well_df['temperature'])
            
            plate_data = ExperimentData(times = plate_measurements_times, temps = plate_temps, plate_name = plate, file_name = file_names[i])

            for row in data_rows:
                for col in data_columns:
                    curr_well = (convert_letter_to_wellkey(row), col)
                    well_key_as_text = row + str(col)

                    # if any well from the replicates was invalid, that is there is an invalid well from the same plate
                    if summary_csvs.loc[(summary_csvs['well'] == well_key_as_text) & (summary_csvs['plate'] == plate) & (summary_csvs['valid'] == 'False')].empty == False:
                        add_line_to_error_log(err_log, f"Well {well_key_as_text} at plate {plate} is invalid and was left out of the graph generation")
                    else:
                        well_summary_data = summary_csvs.loc[(summary_csvs['well'] == well_key_as_text) & (summary_csvs['plate'] == plate) & (summary_csvs['filename'] == file_names[i])].squeeze()
                        # Prep the well data to add to the ExperimentData object
                        well_raw_data = curr_plate_raw_data.loc[(curr_plate_raw_data['well'] == well_key_as_text) &
                                                                (curr_plate_raw_data['plate'] == plate) &
                                                                (curr_plate_raw_data['filename'] == file_names[i])
                                                            ]
                        ODs = list(well_raw_data['OD'])
                        exponent_begin = (well_summary_data["exponent_begin_time"], well_summary_data["exponent_begin_OD"])
                        max_population_gr = (well_summary_data["max_population_gr_time"], well_summary_data["max_population_gr_OD"], well_summary_data["max_population_gr_slope"])
                        exponent_end = (well_summary_data["Time_95%(exp_end)"], well_summary_data["OD_95%"])
                        max_population_density = well_summary_data["max_population_density"]
                        isvalid = well_summary_data["valid"]
                        plate_data.wells[curr_well] = WellData(is_valid = isvalid, ODs = ODs, exponent_begin = exponent_begin, max_population_gr = max_population_gr,
                        exponent_end = exponent_end, max_population_density = max_population_density)
            rep_data.append(plate_data)
            
        parsed_data.append(rep_data)

    # Check that all replicates have the same amount of plates
    if not all(len(rep_data) == len(parsed_data[0]) for rep_data in parsed_data):
        raise ValueError(f'Not all replicates have the same amount of plates in them')
    
    # Trim all lists to the same length
    # Find the min ODs length for the well in all replicates and all the lists should conform to this value
    key = list(parsed_data[0][0].wells)[0]
    min_ODs_len = len(parsed_data[0][0].wells[key].ODs)    
    for i in range(0, len(parsed_data)):
        for key in parsed_data[i][0].wells:
            for plate in parsed_data[i]:
                tmp_len = len(plate.wells[key].ODs)
                if tmp_len < min_ODs_len:
                    min_ODs_len = tmp_len
    
    # Use the value to trim all the lists to the same length
    for replicate_data in parsed_data:
        for plate in replicate_data:
            plate.times = plate.times[:min_ODs_len]
            plate.temps = plate.temps[:min_ODs_len]
            for key in plate.wells:
                plate.wells[key].ODs = plate.wells[key].ODs[:min_ODs_len]

    return parsed_data

def create_single_well_graphs(data, output_path, title, err_log, decimal_percision, draw_95=True):
    '''Create graphs from the data collected in previous steps
    Parameters
    ----------
    data : ExperimentData object
        All the data from the expriment
    output_path : str
        path to the file into which the graphes will be saved to
    title: str
        The title for the graphs
    err_log: [str]
        a refernce to the list containing all the previosuly logged errors
    decimal_percision: int
        The amount of digits after the decimal point to show in the labels
    Returns
    -------
    null

    Examples
    --------   
    >>> create_graphs(parsed_data, "Root/Folder/where_we_want_grpahs_to_be_saved_into")    
    '''
    
    # Styles
    point_size = 50
    alpha = 0.6

    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate
        for key in experiment_data.wells:
            
            try:
                # Setup axis and the figure objects
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.set_xlabel('Time [hours]')
                ax.set_ylabel('OD600')

                # Plot the main graph
                ax.plot(experiment_data.times, experiment_data.wells[key].ODs)

                # If the well is valid graph it with the data from the fitting procedure, otherwise only graph time vs OD as an aid for seeing what went wrong
                curr_well = experiment_data.wells[key]
                if curr_well.is_valid:
                    # Max OD plotting
                    max_OD = experiment_data.wells[key].max_population_density
                    ax.axhline(y=max_OD, color='black', linestyle=':', label=f'Carrying capacity: {str(round(max_OD, decimal_percision))}')

                    # End of lag phase plotting
                    exponent_begin_time, exponent_begin_OD = experiment_data.wells[key].exponent_begin
                    # plot the point with the label
                    ax.scatter([exponent_begin_time], [exponent_begin_OD], s=point_size ,alpha=alpha, 
                                label= f'end of leg phase: {str(round(exponent_begin_time, decimal_percision))} hours')

                    # End of exponential phase
                    if draw_95:
                        time_95, OD_95 = experiment_data.wells[key].exponent_end
                        # plot the point with the label
                        ax.scatter([time_95], [OD_95], c=["darkgreen"], s=point_size ,alpha=alpha,
                                    label=f'95% of growth: {str(round(time_95, decimal_percision))} hours')                        

                    # Max population growth rate plotting
                    x, y, slope = experiment_data.wells[key].max_population_gr
                    # Plot the point and the linear function matching the max population growth rate
                    ax.axline((x, y), slope=slope, color='firebrick', linestyle=':', label=f'maximum population growth rate: {str(round(slope, decimal_percision))}')
                    # plot the point on the graph at which this occures
                    ax.scatter([x], [y], c=['firebrick'], s=point_size, alpha=alpha)

                    ax.legend(loc="lower right")
                
                # Save the figure
                fig.savefig(os.path.join(output_path, f"well {convert_wellkey_to_text(key)} from {experiment_data.plate_name} in {experiment_data.file_name}"))
                
            except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log,
                f"Graphing of cell {convert_wellkey_to_text(key)} at plate: {experiment_data.plate_name} failed with the following exception mesaage: {str(e)}")
            finally:
                plt.close('all')

def create_reps_avarage_graphs(reps, averaged_reps, output_path):
    '''
    Create graphs from the data collected in previous steps
    Parameters
    ----------
    reps : [ExperimentData object]
        All the data from the expriment
    averaged_rep : ExperimentData object
        The data from the expriment after the averaging procedure
    output_path : str
        path to the file into which the graphes will be saved to
    Returns
    -------
    null
    '''
    # Iterate over all the plates
    for i in range(0, len(reps[0])):
        # Loop over all wells within each plate
        for key in averaged_reps[i].wells:
            # Setup axis and the figure objects
            fig, ax = plt.subplots()
            ax.set_title(f"Average of wells {convert_wellkey_to_text(key)} from all wells from {reps[0][i].plate_name}")
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('OD600')

            # Plot the main graph
            ax.plot(averaged_reps[i].times[0], averaged_reps[i].wells[key].ODs, color='black', label='Average')
            # Plot the replicates graphs
            for j in range(0, len(reps)):
                ax.plot(reps[j][i].times, reps[j][i].wells[key].ODs, linestyle=':', label=f'{reps[j][i].plate_name} from replicate {i+j+1}')
            
            
            ax.legend(loc="lower right")
            # Save the figure
            fig.savefig(os.path.join(output_path, f"Average of wells {convert_wellkey_to_text(key)} in {reps[j][i].plate_name}"))
            plt.close('all')

def create_data_tables(experiment_data, output_path, err_log):
    '''Create tables from the data collected in previous steps.
    One table will contain all the data we measured during the experiment in an idetifiable way.
    Fields: filename, plate, well, time, OD, temperature.
    This table will be refered to as 'raw_data'.

    The second table will contain the results of the calculations made for each well. 
    Fields: filename, plate, well, exponent_begin_time, exponent_begin_OD, exponent_end_time, exponent_end_OD,
                   max_population_gr_time, max_population_gr_OD, max_population_gr_slope, max_population_density
    Thia table will be refered to as 'wells_summary'
    
    Parameters
    ----------
    data : ExperimentData object
        All the data from the expriment
    output_path : str
        The path to save the result into
    err_log: [str]
        a refernce to the list containing all the previosuly logged errors
    Returns
    -------
    tuple in the structure of: (df_raw_data, df_wells_summary)    
    '''
    try:
        # datasets for the final result to be saved into
        df_raw_data = None
        df_wells_summary = None
        
        # lists to hold all the data before final storage in dataframes
        # raw_data lists
        # filename will be set at the end since it will be the same for all rows
        raw_data_filename = []
        raw_data_plate_names = []
        raw_data_wells = []
        times = []
        ODs = []
        temperatures = []

        # Copy the data to the needed format
        for experiment_data_point in experiment_data:
        # Loop all ODs within each plate
            for key in experiment_data_point.wells:
                key_as_well = convert_wellkey_to_text(key)
                for i, OD in enumerate(experiment_data_point.wells[key].ODs):
                    raw_data_filename.append(experiment_data_point.file_name)
                    raw_data_plate_names.append(experiment_data_point.plate_name.lower())
                    raw_data_wells.append(key_as_well)
                    times.append(experiment_data_point.times[i])
                    ODs.append(OD)
                    temperatures.append(experiment_data_point.temps[i])
                
        df_raw_data = pd.DataFrame(data = {'filename': raw_data_filename, 'plate': raw_data_plate_names, 'well': raw_data_wells, 'time': times, 'OD': ODs, 'temperature': temperatures})
        
        # wells_summary lists
        wells_summary_filename = []
        wells_summary_validity = []
        wells_summary_plate_names = []
        wells_summary_wells = []
        exponent_begin_time = []
        exponent_begin_OD = []
        max_population_density = []
        max_population_density_95_Time = []
        max_population_95_ODs = []
        max_population_gr_time = []
        max_population_gr_OD = []
        max_population_gr_slope = []
        

        # Copy the data to the needed format
        for experiment_data_point in experiment_data:
        # Loop all ODs within each plate
            for key in experiment_data_point.wells:
                try:
                    # Save for reuse
                    curr_well = experiment_data_point.wells[key]

                    wells_summary_filename.append(experiment_data_point.file_name)
                    wells_summary_validity.append(curr_well.is_valid)
                    wells_summary_plate_names.append(experiment_data_point.plate_name)
                    
                    key_as_well = convert_wellkey_to_text(key)
                    wells_summary_wells.append(key_as_well)

                    if curr_well.is_valid:
                        (end_of_Lag_time, end_of_Lag_OD) = curr_well.exponent_begin
                        exponent_begin_time.append(end_of_Lag_time)
                        exponent_begin_OD.append(end_of_Lag_OD)
                        max_population_density.append(curr_well.max_population_density)

                        max_population_95_Time, max_population_95_OD = curr_well.exponent_end
                        max_population_density_95_Time.append(max_population_95_Time)
                        max_population_95_ODs.append(max_population_95_OD)

                        (time, OD, slope) = curr_well.max_population_gr
                        max_population_gr_time.append(time)
                        max_population_gr_OD.append(OD)
                        max_population_gr_slope.append(slope)
                    else:
                        exponent_begin_time.append(-1)
                        exponent_begin_OD.append(-1)
                        max_population_density.append(-1)
                        max_population_density_95_Time.append(-1)
                        max_population_95_ODs.append(-1)
                        max_population_gr_time.append(-1)
                        max_population_gr_OD.append(-1)
                        max_population_gr_slope.append(-1)


                except Exception as e:
                    print(str(e))
                    add_line_to_error_log(err_log,
                    f"Data frame row {convert_wellkey_to_text(key)} at plate: {experiment_data[0].plate_name} failed with the following exception mesaage: {str(e)}")


        df_wells_summary = pd.DataFrame(data = {
                                                    'filename': wells_summary_filename,
                                                    'valid': wells_summary_validity,
                                                    'plate': wells_summary_plate_names,
                                                    'well': wells_summary_wells,
                                                    'exponent_begin_time': exponent_begin_time,
                                                    'exponent_begin_OD': exponent_begin_OD,
                                                    'max_population_density': max_population_density,
                                                    'Time_95%(exp_end)': max_population_density_95_Time,
                                                    'OD_95%': max_population_95_ODs, 
                                                    'max_population_gr_time': max_population_gr_time,
                                                    'max_population_gr_OD': max_population_gr_OD,
                                                    'max_population_gr_slope': max_population_gr_slope
                                                }
                                        )
        
    except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log,
                f"Creation of data tables had an error at plate: {experiment_data.plate_name} it failed with the following exception mesaage: {str(e)}")
    
    return (df_raw_data, df_wells_summary)

#Fitting
def fill_growth_parameters(data, err_log):
    '''
    Train a model and fill fields in the ExperimentData list given
    Parameters
    ----------
    data : ExperimentData object
        All the data from the expriment
    Returns
    -------
    null

    Examples
    --------   
    >>> get_growth_parameters(parsed_data, err_log)    
    '''
    tidy_df_list = create_tidy_dataframe_list(data)

    # Use to retrive the needed data in the previosly created dataframe
    plate_num = 0
    models_trained = 1
    models_to_train = len(data) * 60

    # Loop all plates
    for experiment_data in data:

        utils.clear_console()
        print(f"Started training in {experiment_data.plate_name}")

        # Loop all ODs within each plate and train model
        for key in experiment_data.wells:
            try:
                # Progress indicator
                print()
                print(f"Started training model: {str(models_trained)} out of: {str(models_to_train)}")
                
                # keep the refrence to the current well for reuse
                curr_well = experiment_data.wells[key]

                # Fit a function with a lag phase to the data
                current_lag_model = curveball.models.fit_model(tidy_df_list[plate_num][key], PLOT=False)
                
                # Find the length of the lag phase (also the begining of the exponent phase) using the previously fitted functions
                exponent_begin_time = curveball.models.find_lag(current_lag_model[0])
                exponent_begin_OD = np.interp(exponent_begin_time, experiment_data.times, curr_well.ODs)

                # Save the carrying capacity of the population as determined by the model
                max_population_density = current_lag_model[0].init_params["K"].value
                
                # 95% of growth as an indication of the end of the rapid growth phase
                max_population_density_95 = max_population_density * 0.95
                # Find the first time at which the ovserved OD values exceeded 95%
                max_population_density_95_time = experiment_data.times[np.searchsorted(curr_well.ODs, max_population_density_95).T]
                
                # prep for saving
                exponet_end = (max_population_density_95_time, max_population_density_95)
                
                # Max slope calculation
                # Get the time and OD of the point with the max slope
                t1, y1, max_slope, t2, y2, mu = curveball.models.find_max_growth(current_lag_model[0])                

                # Save model estimations to fields in the object
                curr_well.exponent_begin = (exponent_begin_time, exponent_begin_OD)
                curr_well.max_population_gr = (t1, y1, max_slope)
                curr_well.exponent_end = exponet_end
                curr_well.max_population_density= max_population_density

                curr_well.is_valid = True

                models_trained += 1
                
            except Exception as e:
                print(str(e))
                #f"Fitting of cell {convert_wellkey_to_text(key)} at plate: {experiment_data.plate_name} failed with the following exception mesaage: {str(e)}")
    
        plate_num += 1

# QA fuctions
def get_reps_variation_data(reps_data):
    '''Get a pandas dataframe with the data of the variations between reps
    
     Parameters
    ----------
    reps_data : [ExperimentData]
    
    err_log : [str]
        a refernce to the list containing all the previosuly logged errors
    '''
    data = []

    # Generate the indexes for the pairwise CC test
    indexes = itertools.combinations(range(0, len(reps_data)), 2)

    # Get the amount of time in hours between each two measurement
    # technical repeats run on the same program in the stacker and therefore will have the same gaps between two measurments
    # take the last time and devide it by the amount of measurements done that is the length of the time array
    time_gap_hours_between_measurements = reps_data[0][0].times[-1] / len(reps_data[0][0].times)

    # Check if the reps are close enough to one another to average
    # Run a cross-correlation test pair-wise
    for i1, i2 in indexes:
        for j in range(0, len(reps_data[0])):
            for key in reps_data[i1][j].wells:
                ODs1 = reps_data[i1][j].wells[key].ODs
                ODs2 = reps_data[i2][j].wells[key].ODs

                # run CC on OD1 and OD2 with itself to get a value to normalize against
                perfect_CC_score = max(max(signal.correlate(ODs2, ODs2)), max(signal.correlate(ODs1, ODs1)))

                # Run the CC test and save the results
                # results with indexes toward the middle of the list reflect the score with small shifts
                correlation_res = signal.correlate(ODs1, ODs2)

                # Find the middle index
                middle_index = len(correlation_res) // 2
                middle_CC_score = correlation_res[middle_index]

                max_CC_score_index = np.argmax(correlation_res)

                max_CC_score = correlation_res[max_CC_score_index]
                max_CC_shift_from_mid = (max_CC_score_index - middle_index) * time_gap_hours_between_measurements

                repA = reps_data[i1][j]
                repB = reps_data[i2][j]

                data.append(
                    {
                        'repA': repA.file_name,
                        'repB': repB.file_name,
                        'plate': repA.plate_name,
                        'well': convert_wellkey_to_text(key),
                        'relative_CC_score' : middle_CC_score / perfect_CC_score,
                        'repA_exponent_begin_time': repA.wells[key].exponent_begin[0],
                        'repB_exponent_begin_time': repB.wells[key].exponent_begin[0],
                        'repA_exponent_begin_OD': repA.wells[key].exponent_begin[1],
                        'repB_exponent_begin_OD': repB.wells[key].exponent_begin[1],
                        'repA_max_population_density': repA.wells[key].max_population_density,
                        'repA_max_population_density': repB.wells[key].max_population_density,
                        'repA_Time_95%(exp_end)': repA.wells[key].exponent_end[0],
                        'repB_Time_95%(exp_end)': repB.wells[key].exponent_end[0],
                        'repA_OD_95%': repA.wells[key].exponent_end[1], 
                        'repB_OD_95%': repB.wells[key].exponent_end[1],
                        'repA_max_population_gr_time': repA.wells[key].max_population_gr[0],
                        'repB_max_population_gr_time': repB.wells[key].max_population_gr[0],
                        'repA_max_population_gr_OD': repA.wells[key].max_population_gr[1],
                        'repB_max_population_gr_OD': repB.wells[key].max_population_gr[1],
                        'repA_max_population_gr_slope': repA.wells[key].max_population_gr[2],
                        'repB_max_population_gr_slope': repA.wells[key].max_population_gr[2],

                        'CC_score': middle_CC_score,
                        'max_CC_score' : max_CC_score,
                        'max_CC_score_shift_in_hours' : max_CC_shift_from_mid,
                        'upper_limit_CC_score': perfect_CC_score,
                    }
                )
    return pd.DataFrame(data)

def get_averaged_ExperimentData(reps_data):

    result = []

    all_times = []
    all_temps = []
    

    # average out all the times and temperatures
    for plate_index in range(0, len(reps_data[0])):
        result.append(ExperimentData(plate_name=f'Averaged plate {plate_index + 1}', file_name=reps_data[0][0].file_name))

        # Create a new list to hold the internal reps data. Each element in the lists is the data about the plate from all the reps
        all_times.append([])
        all_temps.append([])

        # add the data from each rep to the list
        for rep in reps_data:
            all_times[-1].append(np.array(rep[plate_index].times))
            all_temps[-1].append(np.array(rep[plate_index].temps))
            
        
        # average the data from each well and save it into the result object
        # for each key - well in the wells dictionary
        for key in reps_data[0][0].wells:
            tmp_ODs = []
            for rep in reps_data:
                tmp_ODs.append(np.array(rep[plate_index].wells[key].ODs))
            
            # save the average of the data from each rep into the well under the appropraite key in result object
            result[-1].wells[key] = WellData(ODs=np.mean(tmp_ODs, axis=0), is_valid=True)
        
        # avarage out all the internal values from the nested lists and put the mean into the mean list
        result[-1].times = np.mean(all_times, axis=1).tolist()
        result[-1].temps = np.mean(all_temps, axis=1).tolist()

    return result

# TODO: Add a check for the CC score of each well to make sure it is not too low and add to code the output into the error log
def flag_invalid_replicates(reps_data):
    '''
    Finds the wells within replicates that are invalid and returns a list of the indexes of the invalid wells
    '''
    invalid_replicates = []
    for i in range(0, len(reps_data)):
        for j in range(0, len(reps_data[i])):
            for key in reps_data[i][j].wells:
                if reps_data[i][j].wells[key].is_valid == False:
                    invalid_replicates.append((i, j, key))
    return invalid_replicates

#Utils


def create_tidy_dataframe_list(data):
    '''Creates a tidy complient pandas dataset that will later by analyzed by curveball.

    Parameters
    ----------
    data : ExperimentData object
        All the data from the expriment

    Returns
    -------
    list[{(int, int) : pandas.DataFrame}]

    Examples
    --------   
    >>> tidy_df_list = create_tidy_dataframe_list(parsed_data)
    '''
    # List of dictionaries of dataframes with a key of (row_index, column_index) for each well in each plate
    result = []

    # Loop through all plates and collect the data in a way that is analyzeable by curveball down the line
    for experiment_data in data:
        # Create an empty dictionary to hold 
        result.append({})
        for key in experiment_data.wells:
            # Create the dictionary that will be converted to the dataframe
            d = {'Time': experiment_data.times, 'OD': experiment_data.wells[key].ODs, 'Temp[°C]': experiment_data.temps}
            result[-1][key] = pd.DataFrame(data=d)

    return result

def convert_letter_to_wellkey(letter):
    return ord(letter) - 66

def convert_wellkey_to_text(key):
    '''Converts a tuple of type (int,int) back to the appropriate human readable index 

    Parameters
    ----------
    key : (int,int)
        The key index of the well location

    Returns
    -------
    str
    '''
    return chr(key[0] + 66) + str(key[1])

def save_err_log(path, name, err_log):
    with open(path + "/" + name + ".txt", 'w') as file:
        file.writelines("% s\n" % line for line in err_log)

if __name__ == "__main__":
    main()