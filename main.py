import os
import pathlib
import curveball
import matplotlib
import numpy as np
import pandas as pd
from well_data import WellData
import matplotlib.pyplot as plt
from experiment_data import ExperimentData


def main():
    # Base config and program parametrs
    base_path = os.path.normcase("c:\Data\\bio-graphs")
    # Input directory
    input_directory = os.path.join(base_path, "In")
    # The directory into which all the graphs will be saved
    output_directory = os.path.join(base_path, "Out")
    
    # Globals
    global ZERO_SUB
    ZERO_SUB = 0.000001

    global LEFT_OFFSET
    LEFT_OFFSET = 1


    # Matplotlib backend mode - a non-interactive backend that can only write to files
    # Before changing to this mode the program would crash on large runs
    matplotlib.use("Agg")

    # Stores all the error messages for logging
    err_log = []
    # The amount of digits after the decimal point to show in plots
    decimal_percision_in_plots = 3

    # Tuple with the all the extensions all the data files
    extensions = (".xlsx",)
    try:
        # Get the data from the files
        # Full run
        #parsed_data = read_data(input_directory, extensions, err_log)
        # Test run
        parsed_data = read_data(input_directory, extensions, err_log, ["B"])
        # Small test run
        #parsed_data = read_data(input_directory, extensions, err_log, ["B"], [7]) # plate 4
        #parsed_data = read_data(input_directory, extensions, err_log, ["B"], [2])
        
        # Analysis of the data
        fill_growth_parameters(parsed_data, err_log)

        # Graph the data and save the figures to the output_directory
        create_graphs(parsed_data, output_directory, "Foo Bar", err_log, decimal_percision_in_plots, True)

        df_raw_data, df_wells_summary = create_data_tables(parsed_data, output_directory, err_log)

        df_raw_data.to_csv(os.path.join(output_directory, f'{parsed_data[0].file_name}_raw_data.csv'), index=False, encoding='utf-8')
        df_wells_summary.to_csv(os.path.join(output_directory, f'{parsed_data[0].file_name}_summary.csv'), index=False, encoding='utf-8')

        save_err_log(output_directory, "Error log", err_log)

    except Exception as e:
        print(str(e))
        add_line_to_error_log(err_log, str(e))
        save_err_log(output_directory, "Error log", err_log)

def read_data(input_directory, extensions, err_log, data_rows=["B", "C", "D" ,"E", "F", "G"], data_columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]):
    '''Read all the data from the files with the given extension in the input directory given
    
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
                    parsed_data.append(ExperimentData([], [], sheet, curr_file_name, {}))

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
                                    parsed_data[-1].wells[curr_well] = WellData(False, [row[i]], (), (), (), 0, [])
                                # There is a previous reading for this cell, therefore normalize it against the first read then save it
                                else:
                                    if row[i] == "OVER":
                                        raise ValueError(f'a measurement with the value of OVER is in cell {str(((row[1]), j + LEFT_OFFSET))} at sheet: {sheet} please fix and try again')
                                    else:
                                        parsed_data[-1].wells[curr_well].ODs.append(row[i] - parsed_data[-1].wells[curr_well].ODs[0])
                except Exception as e:
                    print(str(e))
                    add_line_to_error_log(err_log,
                    f"data read at sheet {sheet} at file {curr_file_name} failed with the following exception mesaage: {str(e)}")
    # Zero out all the first ODs since normalization was done in relation to them and it's finished, therefore they need to be set to 0
    for experiment_data in parsed_data:
        for row_index, column_index in experiment_data.wells:
            experiment_data.wells[(row_index, column_index)].ODs[0] = 0          


    return parsed_data

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

                # Get the maximal obsereved OD
                max_population_density = max(curr_well.ODs)
                
                # X percentile of growth as an indication of the end of the rapid growth phase
                max_population_density_99 = max_population_density * 0.99
                max_population_density_99_time = 0
                
                max_population_density_95 = max_population_density * 0.95
                max_population_density_95_time = 0
                
                # Find the first time at which the ovserved OD values exceeded the X percentile
                for i in range(len(curr_well.ODs)):
                    if curr_well.ODs[i] > max_population_density_95 and max_population_density_95_time == 0:
                        max_population_density_95_time = experiment_data.times[i]
                    elif curr_well.ODs[i] > max_population_density_99 and max_population_density_99_time == 0:
                        max_population_density_99_time = experiment_data.times[i]
                    
                # prep for saving
                exponet_end = {}
                exponet_end[99] = (max_population_density_99_time, max_population_density_99)
                exponet_end[95] = (max_population_density_95_time, max_population_density_95)

                # Finding the end of the rapid growth phase
                t = np.array(experiment_data.times)
                N = np.array(curr_well.ODs)
                t, N = remove_normalization_artifacts(t, N)
                
                # Calculate the expected OD value of a point based on the time it was messured
                ODs_exponent_values = []
                
                # Fit an exponent to the data to get the point in which we get the maximum slope
                # a - slope, b - intercept
                a, b = curveball.models.fit_exponential_growth_phase(t, N, k=2)
                N0 = np.exp(b)
                # Use the fitted exponent to find the matching ODs by time
                for time in t:
                    ODs_exponent_values.append(exponential_phase(time, N0, a))
               
                # Max slope calculation
                # Get the time and OD of the point with the max slope
                t1, y1, max_slope, t2, y2, mu = curveball.models.find_max_growth(current_lag_model[0])                

                # Save model estimations to fields in the object
                curr_well.exponent_begin = (exponent_begin_time, exponent_begin_OD)
                curr_well.max_population_gr = (t1, y1, max_slope)
                curr_well.exponent_end = exponet_end
                curr_well.max_population_density= max_population_density
                curr_well.exponent_ODs = ODs_exponent_values

                curr_well.is_valid = True

                models_trained += 1
                
            except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log,
                 f"Fitting of cell {convert_wellkey_to_text(key)} at plate: {experiment_data.plate_name} failed with the following exception mesaage: {str(e)}")
    
        plate_num += 1
    
def exponential_phase(t, N0, slope):
    '''
    Calculate the value of a point with N0 * e^(slope * t)
    Parameters
    ----------
    N0 : float
        The first OD messured for the well after the end of the lag phase
    slope : float
        the slope of the the power of the exponenet
    t: float
        the time of the messurments in hours
    Returns
    -------
    N0 * e^(slope * t) 
    '''
    return N0 * np.exp(slope * t)

def remove_normalization_artifacts(t, N):
    '''
    Make sure there are no zeros or negative values (that came from the normalization) in the arrray to run log10 on the data
    ----------
    t : [float]
        The times of the experiment
    N : [float]
        OD at time t[i]
    Returns
    -------
    (t, N)
    '''
    # Find the index of the first non negative value in the N array
    first_positive_index = np.searchsorted(N, ZERO_SUB).T

    for i in range(len(t)):
        if t[i] == 0:
            t[i] = ZERO_SUB
        if N[i] <= 0:
            N[i] = N[first_positive_index]
    return (t, N)

def create_graphs(data, output_path, title, err_log, decimal_percision, draw_exponential_phase=False, draw_99=False, draw_95=True, draw_85=False):
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
                    ax.axhline(y=max_OD, color='black', linestyle=':', label=f'maximum OD600: {str(round(max_OD, decimal_percision))}')

                    # End of lag phase plotting
                    exponent_begin_time, exponent_begin_OD = experiment_data.wells[key].exponent_begin
                    # plot the point with the label
                    ax.scatter([exponent_begin_time], [exponent_begin_OD], s=point_size ,alpha=alpha, 
                                label= f'end of leg phase: {str(round(exponent_begin_time, decimal_percision))} hours')

                    # End of exponential phase
                    if draw_95:
                        time_95, OD_95 = experiment_data.wells[key].exponent_end[95]
                        # plot the point with the label
                        ax.scatter([time_95], [OD_95], c=["darkgreen"], s=point_size ,alpha=alpha,
                                    label=f'95% of growth: {str(round(time_95, decimal_percision))} hours')

                    if draw_99:
                        time_99, OD_99 = experiment_data.wells[key].exponent_end[99]
                        # plot the point with the label
                        ax.scatter([time_99], [OD_99], c=["royalblue"], s=point_size ,alpha=alpha,
                                    label=f'99% of growth: {str(round(time_99, decimal_percision))} hours')
                        

                    # Max population growth rate plotting
                    x, y, slope = experiment_data.wells[key].max_population_gr
                    # Plot the point and the linear function matching the max population growth rate
                    ax.axline((x, y), slope=slope, color='firebrick', linestyle=':', label=f'maximum population growth rate: {str(round(slope, decimal_percision))}')
                    # plot the point on the graph at which this occures
                    ax.scatter([x], [y], c=['firebrick'], s=point_size, alpha=alpha)

                    # Exponential growth rate graph
                    if draw_exponential_phase:
                        # Find the index in which the exponent croses the maximum of the original data
                        # + 1 to keep drawing a little more after the exponent croses the max
                        stop_index = np.searchsorted(experiment_data.wells[key].exponent_ODs, max_OD).T + 1
                        ax.plot(experiment_data.times[:stop_index], experiment_data.wells[key].exponent_ODs[:stop_index], alpha=alpha-0.2)

                    ax.legend(loc="lower right")
                
                # Save the figure
                fig.savefig(os.path.join(output_path, f"well {convert_wellkey_to_text(key)} from {experiment_data.plate_name} in {experiment_data.file_name}"))
                
            except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log,
                f"Graphing of cell {convert_wellkey_to_text(key)} at plate: {experiment_data.plate_name} failed with the following exception mesaage: {str(e)}")
            finally:
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
                    raw_data_plate_names.append(experiment_data_point.plate_name)
                    raw_data_wells.append(key_as_well)
                    times.append(experiment_data_point.times[i])
                    ODs.append(OD)
                    temperatures.append(experiment_data_point.temps[i])
                
        df_raw_data = pd.DataFrame(data = {'filename': raw_data_filename, 'plate': raw_data_plate_names, 'well': raw_data_wells, 'time': times, 'OD': ODs, 'temperature': temperatures})
        
        # wells_summary lists
        wells_summary_filename = []
        wells_summary_plate_names = []
        wells_summary_wells = []
        exponent_begin_time = []
        exponent_begin_OD = []
        max_population_density = []
        max_population_density_95_Time = []
        max_population_density_99_Time = []
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

                    # if the well is invalid don't include it in the file
                    if curr_well.is_valid:
                        wells_summary_filename.append(experiment_data_point.file_name)
                        wells_summary_plate_names.append(experiment_data_point.plate_name)
                        
                        key_as_well = convert_wellkey_to_text(key)
                        wells_summary_wells.append(key_as_well)
                        
                        (end_of_Lag_time, end_of_Lag_OD) = curr_well.exponent_begin
                        exponent_begin_time.append(end_of_Lag_time)
                        exponent_begin_OD.append(end_of_Lag_OD)
                        max_population_density.append(curr_well.max_population_density)

                        max_population_percentiles = curr_well.exponent_end
                        # 95% data
                        max_population_95_Time = max_population_percentiles[95][0]
                        max_population_density_95_Time.append(max_population_95_Time)
                        # 99% data
                        max_population_99_Time = max_population_percentiles[99][0]
                        max_population_density_99_Time.append(max_population_99_Time)
                        
                        (time, OD, slope) = curr_well.max_population_gr
                        max_population_gr_time.append(time)
                        max_population_gr_OD.append(OD)
                        max_population_gr_slope.append(slope)
                    else:
                        add_line_to_error_log(err_log,
                        f"Data frame row {convert_wellkey_to_text(key)} at plate: {experiment_data[0].plate_name} is invalid invalid and was left out of the summary file")
                except Exception as e:
                    print(str(e))
                    add_line_to_error_log(err_log,
                    f"Data frame row {convert_wellkey_to_text(key)} at plate: {experiment_data[0].plate_name} failed with the following exception mesaage: {str(e)}")


        df_wells_summary = pd.DataFrame(data = {
                                                    'filename': wells_summary_filename,
                                                    'plate': wells_summary_plate_names,
                                                    'well': wells_summary_wells,
                                                    'exponent_begin_time': exponent_begin_time,
                                                    'exponent_begin_OD': exponent_begin_OD,
                                                    'max_population_density': max_population_density,
                                                    'Time_95%': max_population_density_95_Time,
                                                    'Time_99%': max_population_density_99_Time,
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

def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path ,file))
    return files

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

def add_line_to_error_log(log, new_line):
    log.append(new_line)

def save_err_log(path, name, err_log):
    with open(path + "/" + name + ".txt", 'w') as file:
        file.writelines("% s\n" % line for line in err_log)

if __name__ == "__main__":
    main()