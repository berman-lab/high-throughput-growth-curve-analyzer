import os
import curveball
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
    
    err_log = []

    # Tuple with the all the extensions all the data files
    extensions = (".xlsx")
    
    # Get the data from the files
    parsed_data = read_data(input_directory, extensions, err_log)

    # Analysis of the data using curveball
    get_growth_parameters(parsed_data, err_log)

    # Graph the data and save the figures to the output_directory
    create_graphs(parsed_data, output_directory, "New Title", err_log, False)

    save_err_log(output_directory, "Error log", err_log)

def read_data(input_directory, extensions, err_log):
    '''Read all the data from the files with the given extension in the input directory given
    
     Parameters
    ----------
    input_directory : str
        The path to the folder where all the data we want to analyze is stored
    extensions : (str, str, ...)
        tuple with all the files with a given file extension we wish to include in the analysis

    Returns
    -------
    ExperimentData object

    Examples
    --------   
    >>> read_data("Root/Folder/where_we_want_to_read_data_from", ("xslx", "csv"))
    '''

    # The index the data we want to analyze starts at
    #cutoff_index = 78
    cutoff_index = 40

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
                
                parsed_data.append(ExperimentData({}, [], [], sheet, curr_file_name, {}, {}, {}, {}))

                # Load the current sheet of the excel file
                df = pd.read_excel(excel_file, sheet, header = cutoff_index)

                # run tourgh all the rows and columns and save the data into object for graphing later
                # We use 96 well plates but only use the inner wells. That is, we treat the 96 well as a 60 well (6 X 10)
                for _, row in enumerate(df.itertuples(), 1):
                    # save the time of reading from the start of the experiment in seconds
                    if row[1] == "Time [s]":
                        parsed_data[-1].times.append(row[2] / 3600)
                    # save the tempreture at the time of reading
                    elif row[1] == "Temp. [°C]":
                        parsed_data[-1].temps.append(row[2])
                    # save the OD of the well
                    
                    elif row[1] in ["B", "C", "D" ,"E", "F", "G"]:
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


    # Use to retrive the needed data frame previosly created
    plate_num = 0

    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate and train model
        for row_index, column_index in experiment_data.ODs:
            try:
                key = (row_index, column_index)
                current_model = curveball.models.fit_model(tidy_df_list[plate_num][key], PLOT = False)
                begin_exponent_time = curveball.models.find_lag(current_model[0])
                # max_growth stuff
                # a - maximum population growth rate
                # μ - maximum of the per capita growth rate
                t1, y1, a, t2, y2, mu = curveball.models.find_max_growth(current_model[0])

                # K - maximum population density
                max_population_density = current_model[0].params['K'].value

                # Save model estimations to fields in the object
                experiment_data.begin_exponent_time[key] = begin_exponent_time
                experiment_data.max_population_gr[key] = a
                experiment_data.max_per_capita_gr[key] = mu
                experiment_data.max_population_density[key]= max_population_density
            except Exception as e:
                add_line_to_error_log(err_log, "Fitting of cell " + convert_wellkey_to_text(key) + " at plate: " + experiment_data.plate_name + 
                 " failed with the following exception mesaage: " + str(e))
    
        plate_num += 1   
    
def create_graphs(data, output_path, title, err_log, verbos=False):
    '''Create graphs from the data collected in previous steps
    Parameters
    ----------
    data : ExperimentData object
        All the data from the expriment
    output_path : str
        path to the file into which the graphes will be saved to
    title: str
        The title for the graphs
    verbos: boolean
        Include extra lines along the Y axis to split up X values
    Returns
    -------
    null

    Examples
    --------   
    >>> create_graphs(parsed_data, "Root/Folder/where_we_want_grpahs_to_be_saved_into")    
    '''
    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate
        for row_index, column_index in experiment_data.ODs:
            key = (row_index, column_index)
            
            # Set the first value to 0 since it was used to normalize against
            experiment_data.ODs[key][0] = 0

            # Create the graph and save it
            fig, ax = plt.subplots()

            ax.set_title(title)
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('OD600')

            # Plot the main graph
            ax.plot(experiment_data.times, experiment_data.ODs[key])

            # Check if there are values or if the fitting failed for the cell
            if key in experiment_data.max_population_density and key in experiment_data.begin_exponent_time :
                plt.axhline(y=experiment_data.max_population_density[key], color='black', linestyle='-')
                plt.axvline(x=experiment_data.begin_exponent_time[key], ymin=0, ymax=1, color='black')

                # scalar = 0.025
                # max_OD = experiment_data.ODs[key][-1]
                # exponent_begin_OD = np.interp(experiment_data.begin_exponent_time[key], experiment_data.times, experiment_data.ODs[key])
                # plt.axvline(x=experiment_data.begin_exponent_time[key], ymin=exponent_begin_OD - (max_OD * scalar), ymax=exponent_begin_OD + (max_OD * scalar), color='black')

                # Create a list of values in the best fit line
                #abline_values = [slope * i + intercept for i in x]

                # Plot the best fit line over the actual values
                # plt.plot(x, y, '--')
                # plt.plot(x, abline_values, 'b')
                # plt.title(slope)
            
            
            # ax.legend()

            # Adds more data to help with graph analysis
            if verbos:
                # Add vertical Lines
                for xc in range (1, int(experiment_data.times[-1])):
                    plt.axvline(x=xc)
                y = 0.05
                while y <= max(experiment_data.ODs[(row_index, column_index)]):
                    plt.axhline(y=y, color='r', linestyle='-')
                    y += 0.05

            # Save the figure
            plt.savefig(output_path + "/well " + chr(row_index + 66) + "," + str(column_index + 3) + " from " + experiment_data.file_name + " " + experiment_data.plate_name)
            plt.close()

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
    return chr(key[0] + 66) + "," + str(key[1] + 3)

def add_line_to_error_log(log, new_line):
    log.append(new_line)

def save_err_log(path, name, err_log):
    with open(path + "/" + name + ".txt", 'w') as file:
        file.writelines("% s\n" % line for line in err_log)


if __name__ == "__main__":
    main()