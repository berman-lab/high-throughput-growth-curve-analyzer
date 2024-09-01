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
    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', '--input_folder', help='The path to the folder with the measurements', required=True)
    parser.add_argument('-out', '--output_folder', help='The path to the folder under which the output will be saved', required=True)
    parser.add_argument('-f', '--format', help='The layout of the plates (96, 384)', required=True)
    parser.add_argument('-g', '--is_create_graphs', help='should single well graphs be generated or only tables and summary graphs', default=False ,action='store_true')
    parser.add_argument('-c', '--is_get_cached_results', help='should the program grab data from the output folder that was put there from previous runs', default=False ,action='store_true')
    
    args = parser.parse_args()
    input_path = os.path.normcase(args.input_folder)
    output_path = os.path.normcase(args.output_folder)
    format = int(args.format)
    is_create_graphs = args.is_create_graphs
    is_import_chached_results = args.is_get_cached_results
    

    # Read the data from the config file based on the format provided as an argument
    config = ''
    with open('config.json') as json_file:
        config = json.load(json_file)
    
    plate_rows = config[f'{format}_plate_rows']
    plate_columns = config[f'{format}_plate_columns']
    plate_repeats = config['plate_repeats']
    condition_file_map = config['condition_file_map']
    raw_data_file_extension = config['file_extension']

    # Globals
    # Valued used to replace zeros with a values close to 0
    # that will not cause errors when applaying log to the data
    global ZERO_SUB
    ZERO_SUB = 0.000001

    # The amount of digits after the decimal point to show in plots
    global DECIMAL_PERCISION_IN_PLOTS
    DECIMAL_PERCISION_IN_PLOTS = 1

    if not is_import_chached_results:
        # Get all the files from the input directory
        files_for_analysis = gc_utils.get_files_from_directory(input_path, raw_data_file_extension)

        file_import_log = []
        file_import_log_save_path = os.path.join(output_path, 'file_import_log.txt')
        
        file_raw_data_df_mapping = {}
        print("Importing the data from files")
        # Add all the file names as keys to the dictionary and save the data in the dictionary
        for file in files_for_analysis:
            current_file_name = pathlib.Path(file).stem
            file_raw_data_df_mapping[current_file_name] = gc_io.read_tecan_stacker_xlsx(file, plate_rows, plate_columns, file_import_log)
            # Save the dataframes to a csv file
            gc_io.save_dataframe_to_csv(file_raw_data_df_mapping[current_file_name], output_path, f'{current_file_name}_raw_data')
        print("Exported raw data to csv")

        gc_utils.save_log(file_import_log, file_import_log_save_path)


        growth_parameters_log = []
        growth_parameters_log_save_path = os.path.join(output_path, 'growth_parameters_log.txt')
        print("Calculating growth parameters for each plate and well")
        file_summary_df_mapping = {}
        # Caclulate growth parameters for each experiment
        for file_name in file_raw_data_df_mapping:
            file_summary_df_mapping[file_name], curr_log = gc_core.get_experiment_growth_parameters(file_raw_data_df_mapping[file_name], growth_parameters_log)
            growth_parameters_log += curr_log
            gc_io.save_dataframe_to_csv(file_summary_df_mapping[file_name], output_path, f'{file_name}_summary_data')
        
        # Remove all the empty indexes from the list before saving
        growth_parameters_log = list(filter(lambda x: x != '', growth_parameters_log))
        gc_utils.save_log(growth_parameters_log, growth_parameters_log_save_path)


        figures_log = []
        figures_log_save_path = os.path.join(output_path, 'figures_log.txt')
        if is_create_graphs:
            print("Creating figures")
            # Graph the data and save the figures to the output_directory
            for file_name in file_raw_data_df_mapping:
                well_save_path = f'{file_name} single well graphs'
                gc_io.create_directory(output_path, well_save_path)
                graphs_output_path = os.path.join(output_path, well_save_path)
                gc_io.create_single_well_graphs(file_name, file_raw_data_df_mapping[file_name], file_summary_df_mapping[file_name], graphs_output_path,
                                                "OD600nm as a function of time in hours", DECIMAL_PERCISION_IN_PLOTS)

        gc_utils.save_log(figures_log, figures_log_save_path)

        multiple_well_comparison_log = []
        multiple_well_comparison_log_save_path = os.path.join(output_path, 'multiple_well_comparison_log.txt')
        print("Comparing multiple repeats")
        # Check that the user provided config makes sense
        if len(file_raw_data_df_mapping) == 1 and (plate_repeats == [] or all([len(item) == 1 for item in plate_repeats])):
            err_text = 'No repeats were provided and only one file was provided. No analysis can be done across plates. Finishing the program.'
            multiple_well_comparison_log.append(err_text)
            print(err_text)
            return

        variation_matrix = gc_core.get_reps_variation_data(file_raw_data_df_mapping, file_summary_df_mapping, plate_repeats, condition_file_map, multiple_well_comparison_log)

        variation_matrix_unidexed = variation_matrix.reset_index()
        variation_matrix_unidexed.to_csv(os.path.join(output_path, f'{list(file_raw_data_df_mapping.keys())[0]}-{list(file_raw_data_df_mapping.keys())[-1]}_coupled_reps_data.csv'), index=False, encoding='utf-8')
        
        gc_utils.save_log(multiple_well_comparison_log, multiple_well_comparison_log_save_path)

    else:
        print(f"Importing results in {output_path}")
        file_raw_data_df_mapping, file_summary_df_mapping, variation_matrix = gc_io.import_previous_run_data(output_path)
        print(f"Import succesful, imported {file_raw_data_df_mapping.keys()} raw data and summary data and the multiple files comprison table")


    # The data is in the needed objects, we can use it for the 
    multiple_reps_and_files_summary_result = gc_core.multiple_reps_and_files_summary(condition_file_map, plate_repeats, file_raw_data_df_mapping, file_summary_df_mapping, variation_matrix)

    multiple_reps_and_files_summary_result.to_csv(os.path.join(output_path, 'multiple_reps_and_files_summary_result.csv'), index=False, encoding='utf-8')

    # create_reps_avarage_graphs(raw_data, averaged_rep, output_directory)



if __name__ == "__main__":
    start_time = time.time()
    main()
    passed_time = time.time() - start_time
    # Convert the time to minutes and seconds and print it
    print(f"It took {int(passed_time / 60)} minutes and {int(passed_time % 60)} seconds to run the program")