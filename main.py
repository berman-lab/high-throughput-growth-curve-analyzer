import os
import time
import pathlib
import itertools
import numpy as np
import pandas as pd
from scipy import signal

import gc_io
import gc_core
import gc_utils


def main():
    # Base config and program parametrs
    # MacOS default path
    base_path = "/Users/Shared/Data/bio-graphs"

    # Check if the system is running on windows, if yes then set base_path to the windows default path
    if os.name in ('nt', 'dos'):
        base_path = "c:\Data\\bio-graphs"
    
    # Make path string into a path object
    base_path = os.path.normcase(base_path)
    # Input directory
    input_directory = os.path.join(base_path, "In")
    # Output directory
    output_directory = os.path.join(base_path, "Out")
    
    # Globals
    # Valued used to replace zeros with a values close to 0
    # that will not cause errors when applaying log to the data
    global ZERO_SUB
    ZERO_SUB = 0.000001

    # Used to account for the extrs column that the pandas reading function adds to the data
    global LEFT_OFFSET
    LEFT_OFFSET = 1

    # The amount of digits after the decimal point to show in plots
    global DECIMAL_PERCISION_IN_PLOTS
    DECIMAL_PERCISION_IN_PLOTS = 3 

    # Get all the files from the input directory
    files_for_analysis = gc_utils.get_files_from_directory(input_directory, "xlsx")
    # Create a folder for the current analysis in the output directory based on the first file name and update the output directory variable to the new path
    output_directory = gc_io.create_directory(output_directory, pathlib.Path(files_for_analysis[0]).stem)

    #Logging setup with the output directory as the save location of the log file
    logger = gc_utils.get_logger(output_directory)

    # Crearte a dictionary of all the files with the file name as the key and all the measurements in a data as the value in a dataframe
    file_df_mapping = {}
    print("Importing the data from files")
    # Add all the file names as keys to the dictionary and save the data in the dictionary
    for file in files_for_analysis:
        current_file_name = pathlib.Path(file).stem
        file_df_mapping[current_file_name] = gc_io.read_tecan_stacker_xlsx(file, data_rows=["B", "C", "D" ,"E", "F", "G"], data_columns=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
        # Save the dataframes to a csv file
        gc_io.save_dataframe_to_csv(file_df_mapping[current_file_name], output_directory, f'{current_file_name}_raw_data')

    print("Exported raw data to csv")

    summary_dfs = {}
    # Caclulate growth parameters for each experiment
    for file_name in file_df_mapping:
        summary_dfs[file_name] = gc_core.get_experiment_growth_parameters(file_df_mapping[file_name])
        gc_io.save_dataframe_to_csv(summary_dfs[file_name], output_directory, f'{file_name}_summary_data')

    print("Finished calculating growth parameters")

    print("Creating figures")
    # Graph the data and save the figures to the output_directory
    for file_name in file_df_mapping:
        well_save_path = f'{file_name} single well graphs'
        gc_io.create_directory(output_directory, well_save_path)
        graphs_output_path = os.path.join(output_directory, well_save_path)
        gc_io.create_single_well_graphs(file_name, file_df_mapping[file_name], summary_dfs[file_name], graphs_output_path, "OD600[nm] against Time[hours]", DECIMAL_PERCISION_IN_PLOTS)

    # variation_matrix = get_reps_variation_data(raw_data)
    # variation_matrix.to_csv(os.path.join(output_directory, f'{raw_data[0][0].file_name}_coupled_reps_data.csv'), index=False, encoding='utf-8')
    # averaged_rep = get_averaged_ExperimentData(raw_data)
    # create_reps_avarage_graphs(raw_data, averaged_rep, output_directory)

#IO
def create_reps_avarage_graphs(reps, averaged_reps, output_path):
    '''
    Create graphs from the data collected in previous steps
    Parameters
    ----------
    reps : [ExperimentData object]
        All the data from the expriment
    averaged_rep : ExperimentData object
        The data from the expriment after the averaging procedure
    output_path : str
        path to the file into which the graphes will be saved to
    Returns
    -------
    null
    '''
    # Iterate over all the plates
    for i in range(0, len(reps[0])):
        # Loop over all wells within each plate
        for key in averaged_reps[i].wells:
            # Setup axis and the figure objects
            fig, ax = plt.subplots()
            ax.set_title(f"Average of wells {convert_wellkey_to_text(key)} from all wells from {reps[0][i].plate_name}")
            ax.set_xlabel('Time [hours]')
            ax.set_ylabel('OD600')

            # Plot the main graph
            ax.plot(averaged_reps[i].times[0], averaged_reps[i].wells[key].ODs, color='black', label='Average')
            # Plot the replicates graphs
            for j in range(0, len(reps)):
                ax.plot(reps[j][i].times, reps[j][i].wells[key].ODs, linestyle=':', label=f'{reps[j][i].plate_name} from replicate {i+j+1}')
            
            
            ax.legend(loc="lower right")
            # Save the figure
            fig.savefig(os.path.join(output_path, f"Average of wells {convert_wellkey_to_text(key)} in {reps[j][i].plate_name}"))
            plt.close('all')

# QA fuctions
def get_reps_variation_data(reps_data):
    '''Get a pandas dataframe with the data of the variations between reps
    
     Parameters
    ----------
    reps_data : [ExperimentData]
    
    err_log : [str]
        a refernce to the list containing all the previosuly logged errors
    '''
    data = []

    # Generate the indexes for the pairwise CC test
    indexes = itertools.combinations(range(0, len(reps_data)), 2)

    # Get the amount of time in hours between each two measurement
    # technical repeats run on the same program in the stacker and therefore will have the same gaps between two measurments
    # take the last time and devide it by the amount of measurements done that is the length of the time array
    time_gap_hours_between_measurements = reps_data[0][0].times[-1] / len(reps_data[0][0].times)

    # Check if the reps are close enough to one another to average
    # Run a cross-correlation test pair-wise
    for i1, i2 in indexes:
        for j in range(0, len(reps_data[0])):
            for key in reps_data[i1][j].wells:
                ODs1 = reps_data[i1][j].wells[key].ODs
                ODs2 = reps_data[i2][j].wells[key].ODs

                # run CC on OD1 and OD2 with itself to get a value to normalize against
                perfect_CC_score = max(max(signal.correlate(ODs2, ODs2)), max(signal.correlate(ODs1, ODs1)))

                # Run the CC test and save the results
                # results with indexes toward the middle of the list reflect the score with small shifts
                correlation_res = signal.correlate(ODs1, ODs2)

                # Find the middle index
                middle_index = len(correlation_res) // 2
                middle_CC_score = correlation_res[middle_index]

                max_CC_score_index = np.argmax(correlation_res)

                max_CC_score = correlation_res[max_CC_score_index]
                max_CC_shift_from_mid = (max_CC_score_index - middle_index) * time_gap_hours_between_measurements

                repA = reps_data[i1][j]
                repB = reps_data[i2][j]

                data.append(
                    {
                        'repA': repA.file_name,
                        'repB': repB.file_name,
                        'plate': repA.plate_name,
                        'well': convert_wellkey_to_text(key),
                        'relative_CC_score' : middle_CC_score / perfect_CC_score,
                        'repA_exponent_begin_time': repA.wells[key].exponent_begin[0],
                        'repB_exponent_begin_time': repB.wells[key].exponent_begin[0],
                        'repA_exponent_begin_OD': repA.wells[key].exponent_begin[1],
                        'repB_exponent_begin_OD': repB.wells[key].exponent_begin[1],
                        'repA_max_population_density': repA.wells[key].max_population_density,
                        'repA_max_population_density': repB.wells[key].max_population_density,
                        'repA_Time_95%(exp_end)': repA.wells[key].exponent_end[0],
                        'repB_Time_95%(exp_end)': repB.wells[key].exponent_end[0],
                        'repA_OD_95%': repA.wells[key].exponent_end[1], 
                        'repB_OD_95%': repB.wells[key].exponent_end[1],
                        'repA_max_population_gr_time': repA.wells[key].max_population_gr[0],
                        'repB_max_population_gr_time': repB.wells[key].max_population_gr[0],
                        'repA_max_population_gr_OD': repA.wells[key].max_population_gr[1],
                        'repB_max_population_gr_OD': repB.wells[key].max_population_gr[1],
                        'repA_max_population_gr_slope': repA.wells[key].max_population_gr[2],
                        'repB_max_population_gr_slope': repA.wells[key].max_population_gr[2],

                        'CC_score': middle_CC_score,
                        'max_CC_score' : max_CC_score,
                        'max_CC_score_shift_in_hours' : max_CC_shift_from_mid,
                        'upper_limit_CC_score': perfect_CC_score,
                    }
                )
    return pd.DataFrame(data)

def get_averaged_ExperimentData(reps_data):

    result = []

    all_times = []
    all_temps = []
    

    # average out all the times and temperatures
    for plate_index in range(0, len(reps_data[0])):
        result.append(ExperimentData(plate_name=f'Averaged plate {plate_index + 1}', file_name=reps_data[0][0].file_name))

        # Create a new list to hold the internal reps data. Each element in the lists is the data about the plate from all the reps
        all_times.append([])
        all_temps.append([])

        # add the data from each rep to the list
        for rep in reps_data:
            all_times[-1].append(np.array(rep[plate_index].times))
            all_temps[-1].append(np.array(rep[plate_index].temps))
            
        
        # average the data from each well and save it into the result object
        # for each key - well in the wells dictionary
        for key in reps_data[0][0].wells:
            tmp_ODs = []
            for rep in reps_data:
                tmp_ODs.append(np.array(rep[plate_index].wells[key].ODs))
            
            # save the average of the data from each rep into the well under the appropraite key in result object
            result[-1].wells[key] = WellData(ODs=np.mean(tmp_ODs, axis=0), is_valid=True)
        
        # avarage out all the internal values from the nested lists and put the mean into the mean list
        result[-1].times = np.mean(all_times, axis=1).tolist()
        result[-1].temps = np.mean(all_temps, axis=1).tolist()

    return result

# TODO: Add a check for the CC score of each well to make sure it is not too low and add to code the output into the error log
def flag_invalid_replicates(reps_data):
    '''
    Finds the wells within replicates that are invalid and returns a list of the indexes of the invalid wells
    '''
    invalid_replicates = []
    for i in range(0, len(reps_data)):
        for j in range(0, len(reps_data[i])):
            for key in reps_data[i][j].wells:
                if reps_data[i][j].wells[key].is_valid == False:
                    invalid_replicates.append((i, j, key))
    return invalid_replicates

if __name__ == "__main__":
    start_time = time.time()
    main()
    passed_time = time.time() - start_time
    print(f"It took {passed_time}")
    