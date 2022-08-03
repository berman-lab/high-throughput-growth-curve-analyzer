import os

# gc prefix added to avoid name conflict with other modules
# This file contains all general utility functions

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