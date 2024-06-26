import os
import itertools
import curveball
import numpy as np
import scipy.signal
import pandas as pd
import multiprocessing
import concurrent.futures


import gc_utils

def get_experiment_growth_parameters(raw_data_df, log):
    '''
    Desrciption
    -----------
    Get a dataframe with all the growth parameters for each well in the experiment.
    
    Parameters
    ----------
    raw_data_df : pandas.DataFrame
        A Dataframe containing all OD measurement for the well with the time of measurement as described in the doc of : read_tecan_stacker_xlsx (or other import functions).
    log: list of strings
        A list of strings containing the log messages
    
    Returns
    -------
    pandas.DataFrame
        This dataframe will hold all the data gathered from the trained curveball models that were each fitted to a single well data. Dataframe containing the columns:
        - ``file_name`` (:py:class:`str`): the name of the file the well belongs to.
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
        - ``well_row_index`` the row index of the current well.
        - ``well_column_index`` the column index of the current well.
        - ``well_key`` the name of the well in the format "A1", "B2" etc.
        - ``is_valid`` (:py:class:`bool`): True if the well is has passed fitting, False if there was a problem and the fitting process wan't complete.
        - ``lag_end_time`` (:py:class:`float`, in hours): The time at which the lag phase ended.
        - ``lag_end_OD`` (:py:class:`float`, in AU): The OD at which the lag phase ended.
        - ``max_population_gr_time`` (:py:class:`float`, in hours): The time at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_OD`` (:py:class:`float`, in AU): The OD at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_slope`` (:py:class:`float`): The slope of the maximum population growth rate (abbreviated as gr).
        - ``exponet_end_time`` (:py:class:`float`, in hours): The time at which the exponential phase ended, defined as the time at which 95% of the carrying capcity, K, was first observed.
        - ``exponet_end_OD`` (:py:class:`float`, in AU): The OD at which the exponential phase ended (95% of the carrying capcity, K, first observed).
        - ``carrying_capacity`` (:py:class:`float`, in AU): The OD value determined to be the carrying capacity, K.
    '''
    raw_data_df_unindexed = raw_data_df.reset_index()
    # Get all the file_names, platenames and wellnames from the dataframe to iterate over
    # "file_name", "plate", "well_column_index", "well_row_indexes" are the keys by which the df is indexed
    file_names = raw_data_df_unindexed['file_name'].unique()
    plate_names = raw_data_df_unindexed['plate_name'].unique()
    well_row_indexes = raw_data_df_unindexed['well_row_index'].unique()
    well_column_indexes = raw_data_df_unindexed['well_column_index'].unique()
    
    
    # Genrate a list of all the keys to run _get_well_growth_parameters on using ProcessPoolExecutor
    # A reference to the dataframe is added to the keys list so from it can be used in the function along with the other parameters
    items = itertools.product([raw_data_df], file_names, plate_names, well_row_indexes, well_column_indexes, [log])

    # Get the amount of cores to use for multiprocessing
    cores_for_use = multiprocessing.cpu_count()

    # Run _get_well_growth_parameters in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores_for_use) as executor: 
        results = list(executor.map(_get_well_growth_parameters, items))

    # Convert results to a dataframe
    results_df = pd.DataFrame(results, columns=['file_name', 'plate_name', 'well_row_index', 'well_column_index' ,'is_valid', 'lag_end_time', 'lag_end_OD',
                                                'max_population_gr_time', 'max_population_gr_OD', 'max_population_gr_slope',
                                                'exponet_end_time', 'exponet_end_OD', 'carrying_capacity'])

    # Add the well key to the dataframe
    results_df['well_key'] = results_df.apply(lambda row: f'{gc_utils.convert_row_number_to_letter(row["well_row_index"])}{row["well_column_index"] + 1}', axis=1)

    # Reorder the columns so that the well_name is the third column
    results_df = results_df[['file_name', 'plate_name', 'well_row_index', 'well_column_index', 'well_key', 'is_valid', 'lag_end_time', 'lag_end_OD',
                            'max_population_gr_time', 'max_population_gr_OD', 'max_population_gr_slope',
                            'exponet_end_time', 'exponet_end_OD', 'carrying_capacity']]

    results_df = results_df.set_index(['file_name', 'plate_name', 'well_row_index', 'well_column_index'])
    return results_df.sort_index()


def _get_well_growth_parameters(item):
    '''
    Desrciption
    -----------
    Get the growth pramaters from the curveball model trained on the data for a single well
    
    Parameters
    ----------
    item : array-like object containing:
        at index 0, a data frame : pandas.DataFrame
            A Dataframe containing all OD measurement for the well with the time of measurement . Required columns:
            - ``time`` (:py:class:`float`, in hours)
            - ``OD`` (:py:class:`float`, in AU)
                All other columns are ignored.
        at index 1, file_name : str
            The name of the file that is being analysed.
        at index 2, plate_name : str
            The name of the plate being analysed.
        at index 3, well_column_index : int
            The well column index (0 based).
        at index 4, well_row_index : str
            The well row index (0 based).
        at index 5, log : list of strings
            A list of strings containing the log messages
    
    Returns
    -------
    dictioanry
        Keys:
        - ``file_name`` (:py:class:`str`): the name of the file the well belongs to.
        - ``plate_name`` (:py:class:`str`): the name of the plate being analysed.
        - ``well_row_index`` (:py:class:`int`): the well row index, 0 based count (A = 0, B = 1 ...).
        - ``well_column_index`` (:py:class:`int`): the well column index, 0 based count.
        - ``is_valid`` (:py:class:`bool`): True if the well is has passed fitting, False if there was a problem and the fitting process wan't complete.
        - ``lag_end_time`` (:py:class:`float`, in hours): The time at which the lag phase ended.
        - ``lag_end_OD`` (:py:class:`float`, in AU): The OD at which the lag phase ended.
        - ``max_population_gr_time`` (:py:class:`float`, in hours): The time at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_OD`` (:py:class:`float`, in AU): The OD at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_slope`` (:py:class:`float`): The slope of the maximum population growth rate (abbreviated as gr).
        - ``exponet_end_time`` (:py:class:`float`, in hours): The time at which the exponential phase ended, defined as the time at which 95% of the carrying capcity, K, was first observed.
        - ``exponet_end_OD`` (:py:class:`float`, in AU): The OD at which the exponential phase ended (95% of the carrying capcity, K, first observed).
        - ``carrying_capacity`` (:py:class:`float`, in AU): The OD value determined to be the carrying capacity, K.
    '''
    
    df = item[0]
    file_name = item[1]
    plate_name = item[2]
    well_row_index = int(item[3])
    well_column_index = int(item[4])
    log = item[5]   
    well_data = df.xs((file_name, plate_name, well_row_index, well_column_index), level=["file_name", "plate_name", "well_row_index", "well_column_index"])

    well_valid = True

    well_name = f'{gc_utils.convert_row_number_to_letter(well_row_index)}{well_column_index + 1}'

    # Fit a function with a lag phase to the data
    try:
        # Rename time to Time since curveball expects the column to be named 'Time'
        well_data = well_data.rename(columns={'time': 'Time'})
        y0 = np.min(well_data.OD)
        k = np.max(well_data.OD)
        # No justification for the guess values, they are just a guess.
        # The fitting process will determine the best values for the parameters anyway, those are just to get the process started
        r = 0.1
        nu = 0.1
        q0 = 0.1
        v = 0.1

        guess = {'y0': y0, 'k': k, 'r': r, 'nu': nu, 'q0': q0, 'v': v}

        models = curveball.models.fit_model(well_data, param_guess=guess ,PLOT=False, PRINT=False)
        # Change the name back to time
        well_data = well_data.rename(columns={'Time': 'time'})
    except ValueError as e:
        log.append(f'ValueError: {e} for well: {well_name} on plate: {plate_name} in file: {file_name}')
        well_valid = False
        return __dict_for_return(file_name, plate_name, well_row_index, well_column_index, well_valid, -1, -1, -1, -1, -1, -1, -1, -1)
    # The model with the lowest BIC is the best fitting model and it's at index 0
    best_model = models[0]

    # Find the length of the lag phase (also the begining of the exponent phase) using the previously fitted functions
    lag_end_time = curveball.models.find_lag(best_model)
    # if the exponent_begin_time is nan then it means that there was an issue with the fitting and the well data has a problem and should be ignored
    # and later on examined to see what the problem is by the user

    if np.isnan(lag_end_time):
        lag_end_time = -1
        lag_end_OD = -1
        log.append(f'Exponenet begin time could not be estimated for well: {well_name} on plate: {plate_name} in file: {file_name}')
        well_valid = False
    # if the exponent_begin_time is not nan then it means that the fitting was successful and the well data is valid, retriive the OD at the begining of the exponent phase
    else:
        lag_end_OD = np.interp(lag_end_time, well_data.time, well_data.OD)

    # Save the carrying capacity of the population as determined by the model
    carrying_capacity = best_model.values['K']
    
    # If the value is less thatn 0.1 then the well is invalid since the cells probably didn't grow
    if carrying_capacity < 0.1:
        log.append(f'Carrying capacity is less than 0.1 for well: {well_name} on plate: {plate_name} in file: {file_name}, check the well data to see if no growth occured')
        carrying_capacity = -1
        exponet_end_OD = -1
        exponet_end_time = -1
        well_valid = False
    else:
        # 95% of growth as an indication of the end of the rapid growth phase
        exponet_end_OD = carrying_capacity * 0.95
        # Find the first time at which the ovserved OD values exceeded 95% of the carrying capacity
        exponet_end_index = gc_utils.get_first_index(well_data.OD, lambda item: item >= exponet_end_OD)
        if exponet_end_index is None:
            log.append(f'Exponenet end time could not be estimated for well: {well_name} on plate: {plate_name} in file: {file_name}')
            exponet_end_time = -1
            well_valid = False
        else:
            # Use the index to find the time at which the 95% of the carrying capacity was first observed
            exponet_end_time = well_data.time.iloc[exponet_end_index]
        
    # Max slope calculation
    # Get the time and OD of the point with the max slope
    max_population_gr_time, max_population_gr_OD, max_population_gr_slope, t2, y2, mu = curveball.models.find_max_growth(best_model)                
    if np.isnan([max_population_gr_time, max_population_gr_OD, max_population_gr_slope]).any():
        log.append(f'Max slope could not be estimated for well: {well_name} on plate: {plate_name} in file: {file_name}, this probably means that the cells in the well did not grow')
        max_population_gr_time = -1
        max_population_gr_OD = -1
        max_population_gr_slope = -1        
        well_valid = False
    
    return __dict_for_return(file_name, plate_name, well_row_index, well_column_index, well_valid, lag_end_time, lag_end_OD, max_population_gr_time, max_population_gr_OD, max_population_gr_slope,
                    exponet_end_time, exponet_end_OD, carrying_capacity)


# for return from _get_well_growth_parameters
def __dict_for_return(file_name, plate, well_row_index, well_column_index, is_valid, lag_end_time, lag_end_OD, max_population_gr_time, max_population_gr_OD, max_population_gr_slope,
                 exponet_end_time, exponet_end_OD, carrying_capacity):
    return {
        'file_name': file_name, 'plate_name': plate, 'well_row_index': well_row_index, 'well_column_index': well_column_index, 'is_valid': is_valid,
        'lag_end_time': lag_end_time, 'lag_end_OD': lag_end_OD, 'max_population_gr_time': max_population_gr_time, 'max_population_gr_OD': max_population_gr_OD,
        'max_population_gr_slope': max_population_gr_slope, 'exponet_end_time': exponet_end_time, 'exponet_end_OD': exponet_end_OD, 'carrying_capacity': carrying_capacity
    }


def get_reps_variation_data(reps_raw_data, reps_summary_data, repeats, err_log):
    '''
    Desrciption
    -----------
    Get a dataframe with the cross-correlation scores for each well in the replicates and all it's other growth parameters

    Parameters
    ----------
    reps_raw_data : dict
        A dictionary containing the raw data for each replicate
    reps_summary_data : dict
        A dictionary containing the summary data for each replicate
    repeats : [[str, str, str], [str, str], ...]
        A list of lists containing the names of the replicates to compare. List lengths are not constrained.
    err_log : list of strings
        A list of strings containing the log messages

    Returns
    -------
    pandas.DataFrame
        The summary data for both replicates and the cross-correlation scores for each well in the replicates
    '''

    export_data = []

    reps_raw_data_unindexed = {}
    reps_summary_data_unindexed = {}

    file_names = list(reps_raw_data.keys())

    # Start by removing the index from the dataframes since we need to access the data in the index when saving the data to a csv file
    for key in file_names:
        reps_raw_data_unindexed[key] = reps_raw_data[key].reset_index()
        reps_summary_data_unindexed[key] = reps_summary_data[key].reset_index()
        

    # Get all the unique plate names, well column indexes and well row indexes from the dataframe
    plate_names = reps_raw_data_unindexed[file_names[0]].plate_name.unique()
    well_row_indexes = reps_raw_data_unindexed[file_names[0]].well_row_index.unique()
    well_column_indexes = reps_raw_data_unindexed[file_names[0]].well_column_index.unique()
    

    # Get the number of hours between each two measurement
    # technical repeats run on the same program in the stacker and therefore will have the same gaps between two measurments
    # take the last time and devide it by the amount of measurements done that is the length of the time array
    
    single_well_measurement_number = reps_raw_data[file_names[0]].xs((file_names[0], plate_names[0], well_row_indexes[0], well_column_indexes[0]),
                                                                     level=["file_name", "plate_name", "well_row_index", "well_column_index"]).shape[0]

    # Needed later for CC shift in hours
    time_gap_hours_between_measurements = list(reps_raw_data[file_names[0]].time)[-1] / single_well_measurement_number

    # Generate the indexes for the pairwise CC test
    file_name_pairs = list(itertools.combinations(range(0, len(file_names)), 2))

    # If there is only one file then the list will be empty and to create the later combinations add the file name to the list
    if len(file_name_pairs) == 0:
        file_name_pairs.append((file_names[0], file_names[0]))
    
    repeat_pairs = []
    for repeat_group in repeats:
        repeat_pairs.append(list(itertools.combinations(repeat_group, 2))) 

    # If there are no repeats then the list will be empty and to create the later combinations add the file name to the list
    # since each file is a repeat of itself and therefore compare the same plates to one another
    if len(repeat_pairs) == 0:
        repeat_pairs = list(zip(plate_names, plate_names))

    file_names_repeats_pairs_combinations = []
    for repeat_pair in repeat_pairs:
        file_names_repeats_pairs_combinations.append(list(itertools.product(file_name_pairs, repeat_pair)))


    file_names_repeats_pairs_combinations = list(itertools.chain.from_iterable(file_names_repeats_pairs_combinations))


    # Prepare the well indexes of the well rows and columns
    well_indexes = list(itertools.product(well_row_indexes, well_column_indexes))

    final_repeat_combinations = list(itertools.product(file_names_repeats_pairs_combinations, well_indexes))

    for repeat_condition in final_repeat_combinations:
        file_name_A = repeat_condition[0][0][0]
        file_name_B = repeat_condition[0][0][1]

        plate_name_A = repeat_condition[0][1][0]
        plate_name_B = repeat_condition[0][1][1]

        well_index = repeat_condition[1]
        well_row_index = well_index[0]
        well_column_index = well_index[1]

        # Get the data for the two wells
        well_A_data = reps_raw_data[file_name_A].xs((file_name_A, plate_name_A, well_row_index, well_column_index),
                                                                level=["file_name", "plate_name", "well_row_index", "well_column_index"])
        
        well_A_summary_data = reps_summary_data[file_name_A].xs((file_name_A, plate_name_A, well_row_index, well_column_index),
                                                                level=["file_name", "plate_name", "well_row_index", "well_column_index"])

        well_B_data = reps_raw_data[file_name_B].xs((file_name_B, plate_name_B, well_row_index, well_column_index),
                                                               level=["file_name", "plate_name", "well_row_index", "well_column_index"])
        
        well_B_summary_data = reps_summary_data[file_name_B].xs((file_name_B, plate_name_B, well_row_index, well_column_index),
                                                                level=["file_name", "plate_name", "well_row_index", "well_column_index"])
        

        res = __compare_replicates(well_A_data['OD'], well_B_data['OD'], time_gap_hours_between_measurements)

        export_data.append(
            {
                'file_name_A': file_name_A,
                'file_name_B': file_name_B,
                'plate_name_A': plate_name_A,
                'plate_name_B': plate_name_B,
                'well_row_index': well_row_index,
                'well_column_index': well_column_index,
                'well_key': well_A_data.well_key.iloc[0],
                'is_well_A_valid': well_A_summary_data.is_valid.iloc[0],
                'is_well_B_valid': well_B_summary_data.is_valid.iloc[0],
                'relative_CC_score' : res['relative_CC_score'],

                'well_A_exponent_begin_time': well_A_summary_data.lag_end_time.iloc[0],
                'well_B_exponent_begin_time': well_B_summary_data.lag_end_time.iloc[0],
                'well_A_expontent_begin_OD': well_A_summary_data.lag_end_OD.iloc[0],
                'well_B_expontent_begin_OD': well_B_summary_data.lag_end_OD.iloc[0],
                'well_A_carrying_capacity': well_A_summary_data.carrying_capacity.iloc[0],
                'well_B_carrying_capacity': well_B_summary_data.carrying_capacity.iloc[0],
                'well_A_exponent_end_time': well_A_summary_data.exponet_end_time.iloc[0],
                'well_B_exponent_end_time': well_B_summary_data.exponet_end_time.iloc[0],
                'well_A_exponent_end_OD': well_A_summary_data.exponet_end_OD.iloc[0],
                'well_B_exponent_end_OD': well_B_summary_data.exponet_end_OD.iloc[0],
                'well_A_max_population_growth_rate_time': well_A_summary_data.max_population_gr_time.iloc[0],
                'well_B_max_population_growth_rate_time': well_B_summary_data.max_population_gr_time.iloc[0],
                'well_A_max_population_growth_rate_OD': well_A_summary_data.max_population_gr_OD.iloc[0],
                'well_B_max_population_growth_rate_OD': well_B_summary_data.max_population_gr_OD.iloc[0],
                'well_A_max_population_growth_rate_slope': well_A_summary_data.max_population_gr_slope.iloc[0],
                'well_B_max_population_growth_rate_slope': well_B_summary_data.max_population_gr_slope.iloc[0],

                'CC_score': res['CC_score'],
                'max_CC_score' : res['max_CC_score'],
                'max_CC_score_shift_in_hours' : res['max_CC_score_shift_in_hours'],
                'upper_limit_CC_score': res['upper_limit_CC_score'],
            }
        )

    comparison_df = pd.DataFrame(export_data)
    comparison_df = comparison_df.set_index(['file_name_A', 'file_name_B', 'plate_name_A', 'plate_name_B', 'well_row_index', 'well_column_index'])

    return comparison_df


def __compare_replicates(rep1_data, rep2_data, time_gap_hours_between_measurements):
    # run CC on OD1 and OD2 with itself to get a value to normalize against
    perfect_CC_score = max(max(scipy.signal.correlate(rep1_data, rep1_data, method='fft')), max(scipy.signal.correlate(rep2_data, rep2_data, method='fft')))

    # Run the CC test and save the results
    # results with indexes toward the middle of the list reflect the score with small shifts
    correlation_res = scipy.signal.correlate(rep1_data, rep2_data, method='fft')

    # Find the middle index score
    middle_index = len(correlation_res) // 2
    middle_CC_score = correlation_res[middle_index]

    max_CC_score_index = np.argmax(correlation_res)

    max_CC_score = correlation_res[max_CC_score_index]
    max_CC_shift_from_mid = (max_CC_score_index - middle_index) * time_gap_hours_between_measurements


    relative_CC_score = 0
    # If the perfect CC score is 0 then all the data is both curves was zero, therefore the relative_CC_score also needs to be zero
    if perfect_CC_score != 0:
        relative_CC_score = middle_CC_score / perfect_CC_score

    # Return the results in a dictionary
    return {
        'relative_CC_score' : relative_CC_score,
        'CC_score': middle_CC_score,
        'max_CC_score' : max_CC_score,
        'max_CC_score_shift_in_hours' : max_CC_shift_from_mid,
        'upper_limit_CC_score': perfect_CC_score,
    }


def get_averaged_ExperimentData(reps_data):
    return 1