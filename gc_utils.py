import os
import logging

# gc prefix added to avoid name conflict with other modules
# This file contains all general utility functions


def convert_row_letter_to_number(row_letter):
    letters = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E':4, 'F':5, 'G':6, 'H':7, 'I':8, 'J':9, 'K':10, 'L': 11, 'M':12, 'N':13, 'O':14, 'P':15}
    return letters[row_letter]


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