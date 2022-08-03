import os
import logging
import pathlib
import numpy as np
import pandas as pd


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
        - ``filename`` (:py:class:`str`): the name of the file that is being analysed.
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``time`` (:py:class:`float`, in hours)
        - ``OD`` (:py:class:`float`, in AU)
        - ``temperature`` (:py:class:`float` in celcius)
    '''
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
                                logging.critical(err_msg)
                                raise ValueError(err_msg)
                            else:
                                # Check if a value has alredy been saved for the current well
                                key = f'{row_letter}{current_column}'
                                if not key in initial_ODs:
                                    # Save the initial OD value for the current well to later normalize against
                                    initial_ODs[key] = row[current_column]
                                    # Save the OD of the current well and all other data attached to it in a new index in the data array
                                    # Since the first measurement is considered the "zero" is this experiment then set the OD value as 0
                                    # Row order: filename, plate, well, time, OD, temperature
                                    data[curr_index] = [current_file_name, sheet_name, key, cycle_time_in_hours, 0, cycle_temp]
                                    # Increase the index tracking the current insertion into the numpy array
                                    curr_index += 1
                                else:
                                    # Save the OD of the current well and all other data attached to it in a new index in the data array and noramalize it
                                    # Row order: filename, plate, well, time, OD, temperature
                                    data[curr_index] = [current_file_name, sheet_name, key, cycle_time_in_hours, row[current_column] - initial_ODs[key], cycle_temp]
                                    # Increase the index tracking the current insertion into the numpy array
                                    curr_index += 1
    # Create a dataframe from the numpy array and returrn it
    df = pd.DataFrame(data, columns=["filename", "plate", "well", "time", "OD", "temperature"])
    df = df.astype({ 'filename': str, 'plate': str, 'well': str, 'time': float, 'OD': float, 'temperature': float })
    # Make sure there are no empty rows in the dataframe
    return df.dropna()

def save_dataframe_to_csv(df, output_file_path, file_name):
    '''
    Description
    -----------
    Save a dataframe to a csv file
    
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
    df.to_csv(file_path_with_file_name, index=False)
    print(f"Saved data to file {file_path_with_file_name}")
    return file_path_with_file_name

def create_directory(output_directory, nested_directory_name):
    '''
    Description
    -----------
    Create a directory if it does not exist
    
    Parameters
    ----------
    output_directory : str
        The path to the output directory
    nested_directory_name : str
        The name of the nested directory to be created
    '''
    # Create the output directory path
    new_dir_path = os.path.join(output_directory, nested_directory_name)
    # Create the directory if it does not exist
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path
    