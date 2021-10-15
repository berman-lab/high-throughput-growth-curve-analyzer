import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the the source and parse it

# Get the data sources from the files the user chose
excel_files_locations =  [ "C:/Data/AM Test OD600 and GFP (Modified)_20210912_110258.xlsx" ]

# The index the data we want to analyze starts at
cutoff_index = 78

# The container of the data after parsing but pre proccecing
parsed_data = []

# Loop excel_files_locations list to read all the relevant files
for excel_file_location in excel_files_locations:
    # Take the excel_file_location and use it to initiate an ExcelFile object as the context
    with pd.ExcelFile(excel_file_location) as excel_file:
        # Loop all the sheets in the file
        for sheet in excel_file.sheet_names:
            
            # Create a new object to save data into
            parsed_data.append({ 'ODs': {}, 'times': [], 'temps': [], 'plate_name' : sheet})

            # Load the current sheet of the excel file
            df = pd.read_excel(excel_file, sheet, header = cutoff_index)

            # run tourgh all the rows and columbs and save the data into object for graphing later
            # We use 96 well plates but only use the inner wells. That is, we treat the 96 well as a 60 well (6 X 10)
            for _, row in enumerate(df.itertuples(), 1):

                # save the time of reading from the start of the experiment in seconds
                if row[1] == "Time [s]":
                    parsed_data[-1]['times'].append(row[2])
                # save the tempreture at the time of reading
                elif row[1] == "Temp. [°C]":
                    parsed_data[-1]['temps'].append(row[2])
                # save the OD of the well
                elif row[1] == "B" or row[1] == "C" or row[1] == "D" or row[1] == "E" or row[1] == "F" or row[1] == "G":
                    # Cnvert the character index to numaric index to be used to insert under the desired key in ODs
                    # 66 is the ASCII value of B and afterwards all the values are sequencial
                    row_index = ord(row[1]) - 66
                    # Collect all values from the rows to the appropriate 
                    for i in range(2, 11):
                        # i is the index of the relevant cell within the excel sheet j is the adjusted value to make it zero based index
                        # to ve put into ODs
                        j = i - 2
                        if (row_index, j) not in parsed_data[-1]['ODs']:
                            # Subtract 2 from i to bring it to zero indexed count
                            parsed_data[-1]['ODs'][(row_index, j)] = [row[i]]
                        else:
                            parsed_data[-1]['ODs'][(row_index, j)].append(row[i])

# Creating graphs