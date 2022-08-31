import pandas as pd
import concurrent.futures

def get_experiment_growth_parameters(df):
    '''
    Desrciption
    -----------
    Get a dataframe with all the growth parameters for each well in the experiment.
    
    Parameters
    ----------
    df : pandas.DataFrame
        A Dataframe containing all OD measurement for the well with the time of measurement . columns:
        - ``filename`` (:py:class:`str`): the name of the file that is being analysed. [required]
        - ``plate`` (:py:class:`str`): the name of the plate being analysed. : [required]
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column. [required]
        - ``time`` (:py:class:`float`, in hours): [required]
        - ``OD`` (:py:class:`float`, in AU): [required]
        - ``temperature`` (:py:class:`float`, in celcius): [optional]
    
    Returns
    -------
    pandas.DataFrame
        This dataframe will hold all the data gathered from the trained curveball models that were each fittef to a single well data. Dataframe containing the columns:
        - ``filename`` (:py:class:`str`): the name of the file that is being analysed.
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``is_valid`` (:py:class:`bool`): True if the well is has passed fitting, False if there was a problem and the fitting process wan't complete.
        - ``lag_end_time`` (:py:class:`float`, in hours): The time at which the lag phase ended.
        - ``lag_end_OD`` (:py:class:`float`, in AU): The OD at which the lag phase ended.
        - ``max_population_gr_time`` (:py:class:`float`, in hours): The time at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_OD`` (:py:class:`float`, in AU): The OD at which the maximum population growth rate (abbreviated as gr) was observed.
        - ``max_population_gr_slope`` (:py:class:`float`): The slope of the maximum population growth rate (abbreviated as gr).
        - ``exponet_end_time`` (:py:class:`float`, in hours): The time at which the exponential phase ended, defined
    '''
    df_unindexed = df.reset_index()
    # Get all the filenames, platenames and wellnames from the dataframe to iterate over
    # filename + platename + wellname are the keys by which the df is indexed
    file_names = df_unindexed['filename'].unique()
    plate_names = df_unindexed['plate'].unique()
    well_names = df_unindexed['well'].unique()
    
    # Genrate a list of all the keys to run _get_well_growth_parameters on using ProcessPoolExecutor
    # A reference to the dataframe is added to the keys list so from it can be used in the function along with the other parameters
    keys = []
    for file_name in file_names:
        for plate_name in plate_names:
            for well_name in well_names:
                keys.append((df, file_name, plate_name, well_name))
                    
    # Get all growth parameters for each well asynchronously using ProcessPoolExecutor
    return 1

def _get_well_growth_parameters(df, file_name, plate_name, well_name):
    '''
    Desrciption
    -----------
    Get the growth pramaters from the curveball model trained on the data for a single well
    
    Parameters
    ----------
    df : pandas.DataFrame
        A Dataframe containing all OD measurement for the well with the time of measurement . columns:
        - ``Time`` (:py:class:`float`, in hours)
        - ``OD`` (:py:class:`float`, in AU)
    file_name : str
        The name of the file that is being analysed.
    plate_name : str
        The name of the plate being analysed.
    well_name : str
        The well name, a letter for the row and a number of the column.
    
    Returns
    -------
    pandas.Series
        This series will hold all the data gathered from the trained curveball model paramaters. datapoints:
        - ``filename`` (:py:class:`str`): the name of the file the well belongs to.
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
        - ``max_population_density`` (:py:class:`float`, in AU): The OD value determined to be the carrying capacity, K.
    '''
    well_data = df.xs((file_name, plate_name, well_name), level=['filename', 'plate', 'well'])
    return 1