import os
import curveball
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from experiment_data import ExperimentData

def main():
    # Base config and program parametrs
    base_path = "C:/Data/bio-graphs"
    # Input directory
    input_directory = base_path + "/In"
    # The directory into which all the graphs will be saved
    output_directory = base_path + "/Out"
    
    # Matplotlib backend mode - a non-interactive backend that can only write to files
    matplotlib.use("Agg")

    # Stores all the error messages for logging
    err_log = []
    # The amount of digits after the decimal point to show in plots
    decimal_percision_in_plots = 3

    # Tuple with the all the extensions all the data files
    extensions = (".xlsx")
    try:
        # Get the data from the files
        # Full run
        #parsed_data = read_data(input_directory, extensions, err_log, ["B", "C", "D" ,"E", "F", "G"])
        # Test run    
        parsed_data = read_data(input_directory, extensions, err_log, ["E"])
        
        # Analysis of the data
        get_growth_parameters(parsed_data, err_log)

        # Graph the data and save the figures to the output_directory
        create_graphs(parsed_data, output_directory, "Foo Bar", err_log, decimal_percision_in_plots, True)

        df_raw_data, df_wells_summary = create_data_tables(parsed_data, output_directory,err_log)

        # Check if any of them came out as 'None'
    finally:
        save_err_log(output_directory, "Error log", err_log)

def read_data(input_directory, extensions, err_log, data_rows=["A", "B", "C", "D" ,"E", "F", "G", "H"]):
    '''Read all the data from the files with the given extension in the input directory given
    
     Parameters
    ----------
    input_directory : str
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

    Examples
    --------
    >>> read_data("Root/Folder/where_we_want_to_read_data_from", ("xslx", "csv"), err_log, ["B", "C", "D" ,"E", "F", "G"])
    '''

    # The container of the data after parsing but pre proccecing
    parsed_data = []

    # retrive all the data files by extesions from the In directory
    excel_files_paths = get_files_from_directory(input_directory, extensions)

    # Loop excel_files_locations list to read all the relevant files
    for excel_file_location in excel_files_paths:
        # Take the excel_file_location and use it to initiate an ExcelFile object as the context
        with pd.ExcelFile(excel_file_location) as excel_file:
            # Loop all the sheets in the file
            for sheet in excel_file.sheet_names:
                # Get the name of the current file. The last part of the path then remove the file extension
                curr_file_name = excel_file_location.split('/')[-1].split(".")[0]
                # Create a new object to save data into
                
                parsed_data.append(ExperimentData({}, [], [], sheet, curr_file_name, {}, {}, {}, {}, {}))

                # Load the current sheet of the excel file
                df = pd.read_excel(excel_file, sheet)
                

                # run tourgh all the rows and columns and save the data into object for graphing later
                # We use 96 well plates but only use the inner wells. That is, we treat the 96 well as a 60 well (6 X 10)
                for _, row in enumerate(df.itertuples(), 1):
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
                        left_offset = 3
                        # Collect all values from the columns to ODs
                        for i in range(left_offset, 13):
                            # i is the index of the relevant cell within the excel sheet j is the adjusted value to make it zero based index to be used when saving to ODs
                            j = i - left_offset
                            curr_cell = (row_index, j)
                            if curr_cell not in parsed_data[-1].ODs:
                                parsed_data[-1].ODs[curr_cell] = [row[i]]
                            # There is a previous reading for this cell, therefore normalize it against the first read then save it
                            else:
                                if row[i] == "OVER":
                                    raise ValueError('a measurement with the value of OVER is in cell ' + str(((row[1]), j + left_offset))  + ' at sheet: ' + sheet + ' please fix and try again')
                                else:
                                    parsed_data[-1].ODs[curr_cell].append(row[i] - parsed_data[-1].ODs[curr_cell][0])
    # Zero out all the first ODs since normalization was done in relation to them and it's finished, therefore they need to be set to 0
    for experiment_data in parsed_data:
        for row_index, column_index in experiment_data.ODs:
            experiment_data.ODs[(row_index, column_index)][0] = 0          


    return parsed_data

def get_growth_parameters(data, err_log):
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
    >>> get_growth_parameters(parsed_data)    
    '''
    tidy_df_list = create_tidy_dataframe_list(data)

    # Use to retrive the needed data frame previosly created dataframe
    plate_num = 0

    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate and train model
        for row_index, column_index in experiment_data.ODs:
            try:
                key = (row_index, column_index)
                
                # Fit a function with a lag phase to the data
                current_lag_model = curveball.models.fit_model(tidy_df_list[plate_num][key], PLOT=False, PRINT=False)
                
                # Find the length of the lag phase (also the begining of the exponent phase) using the previously fitted functions
                exponent_begin_time = curveball.models.find_lag(current_lag_model[0])
                exponent_begin_OD = np.interp(exponent_begin_time, experiment_data.times, experiment_data.ODs[key])

                # Get the maximal obsereved OD
                max_population_density = max(experiment_data.ODs[key])
                
                # Finding of the exponential phase
                t = np.array(experiment_data.times)
                N = np.array(experiment_data.ODs[key])
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


                # Fit a polynomial to the data to get the point in which we get the maximum slope
                coefficients = np.polyfit(experiment_data.times, experiment_data.ODs[key], 15)
                fitted_polynomial = np.poly1d(coefficients)
                # Derevite to get all the slopes
                growth_slope = fitted_polynomial.deriv()
                # Run on each value in the range to get it's slope
                slopes = growth_slope([time for time in experiment_data.times if time > exponent_begin_time])

                # Make sure to account for the removed items index with an offset
                filter_compensation_offset = len(experiment_data.times) - len(slopes)

                # Retrive the maximal slope
                max_slope = max(slopes)
                # Retrive the index of the point
                max_slope_index = np.argmax(slopes).T + filter_compensation_offset

                # Get the time and OD of the point
                t1 = experiment_data.times[max_slope_index]
                y1 = experiment_data.ODs[key][max_slope_index]

                # Find the end of the exponent phase
                exponent_end_time = None
                exponent_end_OD = None

                # Start the search from the point after the point in which the maximal slope was observed
                # Look for the point in which the slope has decreased by at least 90 percent
                i = max_slope_index + 1
                while i < len(slopes):
                    # Make sure to refrence the right element in slopes by offsetting
                    if slopes[i - filter_compensation_offset] <= max_slope * 0.1:
                        exponent_end_time = experiment_data.times[i]
                        exponent_end_OD = np.interp(exponent_end_time, experiment_data.times, experiment_data.ODs[key])
                        break
                    i += 1

                # Save model estimations to fields in the object
                experiment_data.exponent_begin[key] = (exponent_begin_time, exponent_begin_OD)
                experiment_data.exponent_end[key] = (exponent_end_time, exponent_end_OD)
                experiment_data.max_population_gr[key] = (t1, y1, max_slope)
                experiment_data.max_population_density[key]= max_population_density
                experiment_data.exponent_ODs[key] = ODs_exponent_values
                
            except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log, "Fitting of cell " + convert_wellkey_to_text(key) + " at plate: " + experiment_data.plate_name + 
                 " failed with the following exception mesaage: " + str(e))
    
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
 
    Examples
    --------  
    >>> exponential_phase(N0, slope, t)    
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
 
    Examples
    --------  
    >>> remove_normalization_artifacts(t, N)
    '''
    i = 0
    zero_sub = 0.000001
    while i < len(t):
        if t[i] == 0:
            t[i] = zero_sub
        if N[i] <= 0:
            N[i] = zero_sub
        i += 1

    return (t, N)

def create_graphs(data, output_path, title, err_log, decimal_percision, draw_exponential_phase=False):
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

    point_size = 50
    alpha = 0.6

    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate
        for row_index, column_index in experiment_data.ODs:

            key = (row_index, column_index)
            
            try:
                # Setup axis and the figure objects
                fig, ax = plt.subplots()
                ax.set_title(title)
                ax.set_xlabel('Time [hours]')
                ax.set_ylabel('OD600')

                # Plot the main graph
                ax.plot(experiment_data.times, experiment_data.ODs[key])

                # If there is a value for the parameter than graph it, otherwise leave it out

                # Max OD plotting
                if key in experiment_data.max_population_density:
                    max_OD = experiment_data.max_population_density[key]
                    plt.axhline(y=max_OD, color='black', linestyle=':', label='maximum OD600: ' + str(round(max_OD, decimal_percision)))

                # End of lag phase plotting
                if key in experiment_data.exponent_begin:
                    exponent_begin_time, exponent_begin_OD = experiment_data.exponent_begin[key]
                    # Create and format the string for the label
                    end_of_lag_str = str(round(exponent_begin_time, decimal_percision)) + ' hours'
                    # plot the point with the label
                    plt.scatter([exponent_begin_time], [exponent_begin_OD], s=point_size ,alpha=alpha, label='end of leg phase: ' + end_of_lag_str)

                # End of exponential phase
                if key in experiment_data.exponent_end:
                    exponent_end_time, exponent_end_OD = experiment_data.exponent_end[key]
                    # Make sure none of them is "None"
                    if exponent_end_time != None and exponent_end_OD != None:
                        # Create and format the string for the label
                        end_of_exponent_str = str(round(exponent_end_time, decimal_percision)) + ' hours'
                        # plot the point with the label
                        plt.scatter([exponent_end_time], [exponent_end_OD], c=["royalblue"], s=point_size ,alpha=alpha, label='end of the exponential phase: ' + end_of_exponent_str)

                # Max population growth rate plotting
                if key in experiment_data.max_population_gr:
                    x, y, slope = experiment_data.max_population_gr[key]
                    # Plot the point and the linear function matching the max population growth rate
                    plt.axline((x, y), slope=slope, color='firebrick', linestyle=':', label='maximum population growth rate:' + str(round(slope, decimal_percision)))
                    # plot the point on the graph at which this occures
                    plt.scatter([x], [y], c=['firebrick'], s=point_size, alpha=alpha)


                if draw_exponential_phase:
                    # Find the index in which the exponent croses the maximum of the original data
                    stop_index = np.searchsorted(experiment_data.exponent_ODs[key], max_OD).T + 2
                    ax.plot(experiment_data.times[:stop_index], experiment_data.exponent_ODs[key][:stop_index])

                plt.legend(loc="lower right")
                # Save the figure
                plt.savefig(output_path + "/well " + chr(row_index + 66) + "," + str(column_index + 3) + " from " + experiment_data.file_name + " " + experiment_data.plate_name)
                
            except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log, "Graphing of cell " + convert_wellkey_to_text(key) + " at plate: " + experiment_data.plate_name + 
                " failed with the following exception mesaage: " + str(e))
            finally:
                plt.close('all')
                continue

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
    tuple in the structure of: (raw_data, wells_summary)

    Examples
    --------   
    >>> create_summary_tables(parsed_data, "Root/Folder/where_we_want_files_to_be_saved_into")    
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

        # Columns for df_wells_summary
        wells_summary_columns = [
            'filename', 'plate, well', 'exponent_begin_time', 'exponent_begin_OD',
            'exponent_end_time', 'exponent_end_OD', 'max_population_gr_time',
            'max_population_gr_OD', 'max_population_gr_slope', 'max_population_density'
        ]
        # wells_summary lists
         # filename will be set at the end since it will be the same for all rows
        filename = []
        wells_summary_plate_names = []
        wells_summary_wells = []
        exponent_begin_time = []
        exponent_begin_OD = []
        exponent_end_time = []
        exponent_end_OD = []
        max_population_gr_time = []
        max_population_gr_OD = []
        max_population_gr_slope = []
        max_population_density = []

        # Copy the data to the needed format
        for experiment_data_point in experiment_data:
        # Loop all ODs within each plate
            for row_index, column_index in experiment_data_point.ODs:
                i = 0
                key = (row_index, column_index)
                key_as_well = convert_wellkey_to_text(key)
                for OD in experiment_data_point.ODs[key]:
                    raw_data_filename.append(experiment_data_point.file_name)
                    raw_data_plate_names.append(experiment_data_point.plate_name)
                    raw_data_wells.append(key_as_well)
                    times.append(experiment_data_point.times[i])
                    ODs.append(OD)
                    temperatures.append(experiment_data_point.temps[i])
                    i += 1
                
        df_raw_data = pd.DataFrame(data = {'filename': raw_data_filename, 'plate': raw_data_plate_names, 'well': raw_data_wells, 'time': times, 'OD': ODs, 'temperature': temperatures})
        print(df_raw_data)
    except Exception as e:
                print(str(e))
                add_line_to_error_log(err_log, "Creation of data tables had an error at plate: " + experiment_data.plate_name + 
                " it failed with the following exception mesaage: " + str(e))
    finally:
        return (df_raw_data, df_wells_summary)

def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    return [ path + "/" + file_name for file_name in list(filter(lambda file: file.endswith(extension) , os.listdir(path))) ]

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
        for row_index, column_index in experiment_data.ODs:
            # Create the dictionary that will be converted to the dataframe
            d = {'Time': experiment_data.times, 'OD': experiment_data.ODs[(row_index, column_index)], 'Temp[°C]': experiment_data.temps}
            result[-1][(row_index, column_index)] = pd.DataFrame(data=d)

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

    Examples
    --------   
    >>> convert_wellkey_to_text(key)
    '''
    return chr(key[0] + 66) + str(key[1] + 3)

def add_line_to_error_log(log, new_line):
    log.append(new_line)

def save_err_log(path, name, err_log):
    with open(path + "/" + name + ".txt", 'w') as file:
        file.writelines("% s\n" % line for line in err_log)

if __name__ == "__main__":
    main()