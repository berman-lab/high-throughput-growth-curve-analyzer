import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    base_path = "C:/Data/bio-graphs"
    # Input directory
    input_directory = base_path + "/In"
    # The directory into which all the graphs will be saved
    output_directory = base_path + "/Out"
    
    # Tuple with the all the extensions all the data files
    extensions = (".xlsx")
    
    # Get the data from the files
    parsed_data = read_data(input_directory, extensions)  

    # Graph the data and save the figures to the output_directory
    create_graphs(parsed_data, output_directory)    


def read_data(input_directory, extensions):
    '''Read all the data from the files with the given extension in the input directory given'''
    # The index the data we want to analyze starts at
    cutoff_index = 78

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
                # TODO change to dataclass?
                parsed_data.append({ 'ODs': {}, 'times': [], 'temps': [], 'plate_name' : sheet, 'file_name': curr_file_name })

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

                        # This offset comes from the fact that in this expiremnt we don't right-most columb and the index given by itertuples
                        left_offset = 3
                        # Collect all values from the columbs to ODs
                        for i in range(left_offset, 12):
                            # i is the index of the relevant cell within the excel sheet j is the adjusted value to make it zero based index to be used when saving to ODs
                            j = i - left_offset
                            curr_cell = (row_index, j)
                            if curr_cell not in parsed_data[-1]['ODs']:
                                parsed_data[-1]['ODs'][curr_cell] = [row[i]]
                            # There is a previous reading for this cell, therefore normalize it against the first read than save it
                            else:
                                parsed_data[-1]['ODs'][curr_cell].append(row[i] - parsed_data[-1]['ODs'][curr_cell][0])

    return parsed_data

def create_graphs(data, output_path):
    # Loop all plates
    for experiment_data in data:
        # Loop all ODs within each plate
        for row_index, columb_index in experiment_data['ODs']:
            # Set the first value to 0 since it was used to normalize against
            experiment_data['ODs'][(row_index, columb_index)][0] = 0

            # Create the graph and save it
            fig, ax = plt.subplots()
            ax.plot(experiment_data['times'], experiment_data['ODs'][(row_index, columb_index)])
            ax.set_xlabel('Time[s]')
            ax.set_ylabel('OD600')
            ax.set_title("ODs")
            plt.savefig(output_path + "/well " + chr(row_index + 66) + "," + str(columb_index + 3) + " from " + experiment_data["file_name"] + " " + experiment_data["plate_name"])
            plt.close()

def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    return [ path + "/" + file_name for file_name in list(filter(lambda file: file.endswith(extension) , os.listdir(path))) ]

if __name__ == "__main__":
    main()