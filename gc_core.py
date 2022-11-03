import os
import curveball
import numpy as np
import pandas as pd
import multiprocessing
import concurrent.futures

import gc_utils

def get_experiment_growth_parameters(df):
    '''
    Desrciption
    -----------
    Get a dataframe with all the growth parameters for each well in the experiment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A Dataframe containing all OD measurement for the well with the time of measurement . columns:
        - ``file_name`` (:py:class:`str`): the name of the file that is being analysed. [required]
        - ``plate`` (:py:class:`str`): the name of the plate being analysed. : [required]
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column. [required]
        - ``time`` (:py:class:`float`, in hours): [required]
        - ``OD`` (:py:class:`float`, in AU): [required]
        - ``temperature`` (:py:class:`float`, in celcius): [optional]
    
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
    df_unindexed = df.reset_index()
    # Get all the file_names, platenames and wellnames from the dataframe to iterate over
    # file_name + platename + wellname are the keys by which the df is indexed
    file_names = df_unindexed['file_name'].unique()
    plate_names = df_unindexed['plate'].unique()
    well_names = df_unindexed['well'].unique()
    
    # Genrate a list of all the keys to run _get_well_growth_parameters on using ProcessPoolExecutor
    # A reference to the dataframe is added to the keys list so from it can be used in the function along with the other parameters
    items = []
    for file_name in file_names:
        for plate_name in plate_names:
            for well_name in well_names:
                items.append((df, file_name, plate_name, well_name))

    # Get the amount of cores to use for multiprocessing
    cores_for_use = multiprocessing.cpu_count()

    #Run _get_well_growth_parameters in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=cores_for_use) as executor: 
        results = list(executor.map(_get_well_growth_parameters, items))

    # # Convert results to a dataframe
    results_df = pd.DataFrame(results, columns=['file_name', 'plate', 'well', 'is_valid', 'lag_end_time', 'lag_end_OD',
                                                'max_population_gr_time', 'max_population_gr_OD', 'max_population_gr_slope',
                                                'exponet_end_time', 'exponet_end_OD', 'carrying_capacity'])

    results_df.set_index(["file_name", "plate", "well"], inplace=True)
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
    at index 3, well_name : str
        The well name, a letter for the row and a number of the column.
    
    Returns
    -------
    dictioanry
        Keys:
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
    logger = gc_utils.get_logger()
    
    df = item[0]
    file_name = item[1]
    plate_name = item[2]
    well_name = item[3]
    well_data = df.xs((file_name, plate_name, well_name), level=['file_name', 'plate', 'well'])

    well_valid = True

    # Fit a function with a lag phase to the data
    models = curveball.models.fit_model(well_data, PLOT=False, PRINT=False)
    # The model with the lowest BIC is the best fitting model and it's at index 0
    best_model = models[0]

    # Find the length of the lag phase (also the begining of the exponent phase) using the previously fitted functions
    lag_end_time = curveball.models.find_lag(best_model)
    # if the exponent_begin_time is nan then it means that there was an issue with the fitting and the well data has a problem and should be ignored
    # and later on examined to see what the problem is by the user
    if np.isnan(lag_end_time):
        lag_end_time = -1
        lag_end_OD = -1
        logger.error(f'Exponenet begin time could not be estimated for well: {well_name} on plate: {plate_name} in file: {file_name}')
        well_valid = False
    # if the exponent_begin_time is not nan then it means that the fitting was successful and the well data is valid, retriive the OD at the begining of the exponent phase
    else:
        lag_end_OD = np.interp(lag_end_time, well_data.Time, well_data.OD)

    # Save the carrying capacity of the population as determined by the model
    carrying_capacity = best_model.values['K']
    
    #current_lag_model[0].find_K_ci()
    # If the value is less thatn 0.1 then the well is invalid since the cells probably didn't grow
    if carrying_capacity < 0.1:
        logger.error(f'Carrying capacity is less than 0.1 for well: {well_name} on plate: {plate_name} in file: {file_name}, check the well data to see if no growth occured')
        carrying_capacity = -1
        exponet_end_OD = -1
        exponet_end_time = -1
        well_valid = False
    elif carrying_capacity >= max(well_data.OD):
        logger.error(f'Carrying capacity was estimated to be: {carrying_capacity} on plate: {plate_name} in file: {file_name}. This carrying capacity is larger than the maximum OD achived. The well might not have reached the stationary phase fully')
        carrying_capacity = -1
        exponet_end_OD = -1
        exponet_end_time = -1
        well_valid = False
    else:
        # 95% of growth as an indication of the end of the rapid growth phase
        exponet_end_OD = carrying_capacity * 0.95
        # Find the first time at which the ovserved OD values exceeded 95% of the carrying capacity
        exponet_end_index = gc_utils.get_first_index(well_data.OD, lambda item: item >= exponet_end_OD)
        # Use the index to find the time at which the 95% of the carrying capacity was first observed
        exponet_end_time = well_data.Time[exponet_end_index]
        
    # Max slope calculation
    # Get the time and OD of the point with the max slope
    max_population_gr_time, max_population_gr_OD, max_population_gr_slope, t2, y2, mu = curveball.models.find_max_growth(best_model)                
    if np.isnan([max_population_gr_time, max_population_gr_OD, max_population_gr_slope]).any():
        logger.error(f'Max slope could not be estimated for well: {well_name} on plate: {plate_name} in file: {file_name}, this probably means that the cells in the well did not grow')
        max_population_gr_time = -1
        max_population_gr_OD = -1
        max_population_gr_slope = -1        
        well_valid = False
    return {'file_name': file_name, 'plate': plate_name, 'well': well_name, 'is_valid': well_valid,
            'lag_end_time': lag_end_time, 'lag_end_OD': lag_end_OD, 'max_population_gr_time': max_population_gr_time, 'max_population_gr_OD': max_population_gr_OD, 'max_population_gr_slope': max_population_gr_slope,
            'exponet_end_time': exponet_end_time, 'exponet_end_OD': exponet_end_OD, 'carrying_capacity': carrying_capacity}