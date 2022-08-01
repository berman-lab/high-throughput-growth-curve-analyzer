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
    Read the content of a xlsx file from a tecan stakcer format
    
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
        - ``Filename`` (:py:class:`str`): the letter corresponding to the well row.
        - ``Plate`` (:py:class:`str`): the number corresponding to the well column.
        - ``Well`` (:py:class:`str`): the well name, usually a letter for the row and a number of the column.
        - ``Time`` (:py:class:`float`, in hours)
        - ``OD600`` (:py:class:`float`, in AU)
        - ``Temperature [Celsius]`` (:py:class:`float`)
    '''
    parsed_data = None
    print(f"Reading data from file {pathlib.Path(file_path).stem}")
    
    with pd.ExcelFile(file_path) as excel_file:
        # Loop all the sheets in the file
        for sheet in excel_file.sheet_names:            
            # Load the current sheet of the excel file
            df = pd.read_excel(excel_file, sheet)
            
            # Get the how many measurements were taken for the plate to calculate how large the np.array should be
            number_of_measurements = gc_utils.get_max_cycle_number(df)
            row_count = (len(data_rows) +1) * (len(data_columns) +1) * number_of_measurements
            # Create a a numpy array to store the data
            data = np.zeros((row_count, len(data_rows) +1))

            # Run tourgh all the rows and columns and save the data into object for graphing later
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
            
    # Zero out all the first ODs since normalization was done in relation to them and it's finished, therefore they need to be set to 0
    for experiment_data in parsed_data:
        for row_index, column_index in experiment_data.wells:
            experiment_data.wells[(row_index, column_index)].ODs[0] = 0

    return parsed_data