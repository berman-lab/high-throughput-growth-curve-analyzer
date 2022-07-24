import logging


# This file contains all IO related functions

def read_tecan_stacker_xlsx(input_directory, data_rows=["A" ,"B", "C", "D" ,"E", "F", "G", "H"], data_columns=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]):
    '''
    Desrciption
    -----------
    Read the content of a xlsx file from a tecan stakcer format
    
    Parameters
    ----------
    input_directory : path object
        The path to the folder where all the data for analysis is stored
    data_rows : list of strings
        The rows of the xlsx file that are to be read
    data_columns : list of integers
        The columns of the xlsx file that are to be read
    err_log : [str]
        a refernce to the list containing all the previosuly logged errors

    Returns
    -------
    data : pandas dataframe
        The dataframe containing the data from the xlsx file
    '''
    print("Reading tecan stakcer xlsx file")