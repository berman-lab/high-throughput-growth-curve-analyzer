import os
import itertools
import curveball
import numpy as np
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
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
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
    plate_names = raw_data_df_unindexed['plate'].unique()
    well_column_indexes = raw_data_df_unindexed['well_column_index'].unique()
    well_row_indexes = raw_data_df_unindexed['well_row_index'].unique()
    
    # Genrate a list of all the keys to run _get_well_growth_parameters on using ProcessPoolExecutor
    # A reference to the dataframe is added to the keys list so from it can be used in the function along with the other parameters
    items = itertools.product([raw_data_df], file_names, plate_names, well_row_indexes, well_column_indexes, [log])

    # results = []
    # for item in items:
    #     results.append(_get_well_growth_parameters(item))

    # Get the amount of cores to use for multiprocessing
    cores_for_use = multiprocessing.cpu_count()

    # Run _get_well_growth_parameters in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores_for_use) as executor: 
        results = list(executor.map(_get_well_growth_parameters, items))

    # Convert results to a dataframe
    results_df = pd.DataFrame(results, columns=['file_name', 'plate', 'well_row_index', 'well_column_index' ,'is_valid', 'lag_end_time', 'lag_end_OD',
                                                'max_population_gr_time', 'max_population_gr_OD', 'max_population_gr_slope',
                                                'exponet_end_time', 'exponet_end_OD', 'carrying_capacity'])

    results_df = results_df.set_index(['file_name', 'plate', 'well_row_index', 'well_column_index'])
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
            - ``Time`` (:py:class:`float`, in hours)
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
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
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
    well_data = df.xs((file_name, plate_name, well_column_index, well_row_index), level=["file_name", "plate", "well_column_index", "well_row_index"])

    well_valid = True

    well_name = f'{gc_utils.convert_row_number_to_letter(well_row_index)}{well_column_index + 1}'

    # Fit a function with a lag phase to the data
    try:
        models = curveball.models.fit_model(well_data, PLOT=False, PRINT=False)
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
        lag_end_OD = np.interp(lag_end_time, well_data.Time, well_data.OD)

    # Save the carrying capacity of the population as determined by the model
    carrying_capacity = best_model.values['K']
    
    # If the value is less thatn 0.1 then the well is invalid since the cells probably didn't grow
    if carrying_capacity < 0.1:
        log.append(f'Carrying capacity is less than 0.1 for well: {well_name} on plate: {plate_name} in file: {file_name}, check the well data to see if no growth occured')
        carrying_capacity = -1
        exponet_end_OD = -1
        exponet_end_time = -1
        well_valid = False
    # TODO: retstes if this is needed
    # elif carrying_capacity >= max(well_data.OD):
    #     log.append(f'Carrying capacity was estimated to be: {carrying_capacity} on plate: {plate_name} in file: {file_name}. This carrying capacity is larger than the maximum OD achived. The well might not have reached the stationary phase fully')
    #     carrying_capacity = -1
    #     exponet_end_OD = -1
    #     exponet_end_time = -1
    #     well_valid = False
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
            exponet_end_time = well_data.Time[exponet_end_index]
        
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
        'file_name': file_name, 'plate': plate, 'well_row_index': well_row_index, 'well_column_index': well_column_index, 'is_valid': is_valid,
        'lag_end_time': lag_end_time, 'lag_end_OD': lag_end_OD, 'max_population_gr_time': max_population_gr_time, 'max_population_gr_OD': max_population_gr_OD,
        'max_population_gr_slope': max_population_gr_slope, 'exponet_end_time': exponet_end_time, 'exponet_end_OD': exponet_end_OD, 'carrying_capacity': carrying_capacity
    }