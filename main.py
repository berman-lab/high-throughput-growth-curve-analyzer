import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():

    # TODO Add retrival of all xslx files under the 'in' dirctory
    excel_files_locations =  [ "C:/Data/bio-graphs/In/AM Test OD600 and GFP (Modified)_20210912_110258.xlsx" ]

    # The directory into which all the graphs will be saved
    output_directory = "C:/Data/bio-graphs/Out"

    parsed_data = read_data(excel_files_locations)  

    create_graphs(parsed_data, output_directory)    


def read_data(excel_files_locations):

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
                # Get the name of the current file
                curr_file_name = excel_file_location.split('/')[-1]
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
        aaa = experiment_data['ODs']
        for row_index, columb_index in experiment_data['ODs']:
            print((row_index, columb_index))

    # TODO put graph creatin into loop
    fig, ax = plt.subplots()
    ax.plot(data[0]['times'], data[0]['ODs'][(0, 0)])
    ax.set_xlabel('Time[s]')
    ax.set_ylabel('OD600')
    ax.set_title("ODs")
    plt.savefig(output_path + "/cell 0,0 from AM Test OD600 and GFP (Modified)_20210912_110258.png")

if __name__ == "__main__":
    main()