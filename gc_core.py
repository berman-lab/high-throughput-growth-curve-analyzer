def get_well_growth_parameters(df):
    '''
     Desrciption
    -----------
    Get the growth pramaters from the curveball model trained on the data
    
    Parameters
    ----------
    df : pandas.DataFrame
        A Dataframe containing all OD measurement for the well with the time of measurement . columns:
        - ``Time`` (:py:class:`float`, in hours)
        - ``OD`` (:py:class:`float`, in AU)
    Returns
    -------
    pandas.DataFrame
        This dataframe will hold all the data gathered from the trained curveball model paramaters. Dataframe containing the columns:
        - ``filename`` (:py:class:`str`): the name of the file that is being analysed.
        - ``plate`` (:py:class:`str`): the name of the plate being analysed.
        - ``well`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``is_valid`` (:py:class:`bool`): True if the well is has passed fitting, False if there was a problem and the fitting process wan't complete.
        - ``lag_end_time`` (:py:class:`float`, in hours): The time at which the lag phase ended.
        - ``lag_end_OD`` (:py:class:`float`, in AU): The OD at which the lag phase ended.
        TODO: Add the other parameters    
    '''
    return 1