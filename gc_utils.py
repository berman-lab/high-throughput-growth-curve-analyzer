import os
import logging

# gc prefix added to avoid name conflict with other modules
# This file contains all general utility functions

# Single instance of the logger program wide
global _gc_logger
_gc_logger = None
def get_logger(output_file_path = ''):
    '''
    Description
    -----------
    Get the logger object

    Parameters
    ----------
    output_file_path : str
        The path to the folder where the log file will be saved
        only provide the path on the first call to this function, after that the path will be ignored since the logger is a singleton and will not allow a change of path
    
    Returns
    -------
    logger : logging.Logger
        The logger object
    '''

    # Check if the logger is already initialized. If it isn't, initialize it and set _gc_logger to the logger object
    global _gc_logger
    if _gc_logger == None and output_file_path != '':
        logger = logging.getLogger('gc')
        logger.filemode = 'w'
        logger.setLevel(logging.WARNING)
        file_handler =logging.FileHandler(os.path.join(output_file_path, "messages.log"))
        logger.addHandler(file_handler)
        _gc_logger = logger

    return _gc_logger

def clear_console():
    command = 'clear'
    if os.name in ('nt', 'dos'):  # If Machine is running on Windows, use cls
        command = 'cls'
    os.system(command)

def get_files_from_directory(path , extension):
    '''Get the full path to each file with the extension specified from the path'''
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(os.path.join(path ,file))
    return files

def get_max_cycle_number(df):
    '''Get the number of cycles the tecan stacker did when reading the data
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe in the tecan stacker format
    '''
    first_column = list(df.iloc[:, 0])
    for i in range(len(first_column) -1, -1, -1):
        if first_column[i] == "Cycle Nr.":
            return df.iloc[i,1]

def get_first_index(iterable, condition=lambda x: True):
    '''Get the index of the first element in an iterable that matches the condition'''
    for i, x in enumerate(iterable):
        if condition(x):
            return i