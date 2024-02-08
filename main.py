import os
import time
import json
import pathlib
import argparse
import itertools
import numpy as np
import pandas as pd
from scipy import signal

import gc_io
import gc_core
import gc_utils


def main():
    log = []


    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_folder', help='The path to the folder with the measurements', required=True)
    parser.add_argument('-out', '--output_folder', help='The path to the folder under which the output will be saved', required=True)
    parser.add_argument('-f', '--format', help='The layout of the plates (96, 384)', required=True)
    
    args = parser.parse_args()
    input_path = os.path.normcase(args.input_folder)
    output_path = os.path.normcase(args.output_folder)
    format = int(args.format)
    

    # Read the data from the config file based on the format provided as an argument
    config = ''
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    plate_rows = config[f'{format}_plate_rows']
    plate_columns = config[f'{format}_plate_columns']
    

    # Globals
    # Valued used to replace zeros with a values close to 0
    # that will not cause errors when applaying log to the data
    global ZERO_SUB
    ZERO_SUB = 0.000001

    # The amount of digits after the decimal point to show in plots
    global DECIMAL_PERCISION_IN_PLOTS
    DECIMAL_PERCISION_IN_PLOTS = 3 

    # Get all the files from the input directory
    files_for_analysis = gc_utils.get_files_from_directory(input_path, "xlsx")


    # Crearte a dictionary of all the files with the file name as the key and all the measurements in a data as the value in a dataframe
    file_df_mapping = {}
    print("Importing the data from files")
    # Add all the file names as keys to the dictionary and save the data in the dictionary
    for file in files_for_analysis:
        current_file_name = pathlib.Path(file).stem
        file_df_mapping[current_file_name] = gc_io.read_tecan_stacker_xlsx(file, plate_rows, plate_columns, log)
        # Save the dataframes to a csv file
        gc_io.save_dataframe_to_csv(file_df_mapping[current_file_name], output_path, f'{current_file_name}_raw_data')
    print("Exported raw data to csv")


    print("Calculating growth parameters")
    summary_dfs = {}
    # Caclulate growth parameters for each experiment
    for file_name in file_df_mapping:
        summary_dfs[file_name] = gc_core.get_experiment_growth_parameters(file_df_mapping[file_name], log)
        gc_io.save_dataframe_to_csv(summary_dfs[file_name], output_path, f'{file_name}_summary_data')


    print("Creating figures")
    # Graph the data and save the figures to the output_directory
    for file_name in file_df_mapping:
        well_save_path = f'{file_name} single well graphs'
        gc_io.create_directory(output_path, well_save_path)
        graphs_output_path = os.path.join(output_path, well_save_path)
        gc_io.create_single_well_graphs(file_name, file_df_mapping[file_name], summary_dfs[file_name], graphs_output_path, "OD600[nm] against Time[hours]", DECIMAL_PERCISION_IN_PLOTS)


    #variation_matrix = gc_core.get_reps_variation_data(raw_data)
    # variation_matrix.to_csv(os.path.join(output_directory, f'{raw_data[0][0].file_name}_coupled_reps_data.csv'), index=False, encoding='utf-8')
    # averaged_rep = get_averaged_ExperimentData(raw_data)
    # create_reps_avarage_graphs(raw_data, averaged_rep, output_directory)



if __name__ == "__main__":
    start_time = time.time()
    main()
    passed_time = time.time() - start_time
    # Convert the time to minutes and seconds and print it
    print(f"It took {int(passed_time / 60)} minutes and {int(passed_time % 60)} seconds to run the program")