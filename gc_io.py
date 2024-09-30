import os
import pathlib
import itertools
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import gc_utils

# gc prefix added to avoid name conflict with other modules
# This file contains all IO related functions

def read_tecan_stacker_xlsx(file_path, plate_rows, plate_columns, log):
    '''
    Desrciption
    -----------
    Read the content of a xlsx file in a tecan stakcer format
    
    Parameters
    ----------
    file_path : str
        The path to the file
    plate_rows : list of strings
        The rows of the plates to read data from
    plate_columns : list of integers
        The columns of the plates to read data from
    log : list
        The list to which the log messages will be appended
    Returns
    -------
    pandas.DataFrame
        dataframe containing the data from the input file indxed by file_name, plate_name, row_index, column_index. Columns:
        - ``file_name`` (:py:class:`str`): the name of the file the data comes from.
        - ``plate_name`` (:py:class:`str`): the name of the plate the data comes from.
        - ``well_key`` (:py:class:`str`): the well name, a letter for the row and a number of the column.
        - ``well_row_index`` (:py:class:`int`): the well row index.
        - ``well_column_index`` (:py:class:`str`): the well column index.
        - ``time`` (:py:class:`float`) measurement time in hours
        - ``temperature`` (:py:class:`float`) degrees in celsius
        - ``OD`` (:py:class:`float`) Optical density in AU
    '''

    current_file_name = pathlib.Path(file_path).stem
    print(f"Reading data from file {current_file_name}")

    file_names = []
    plate_names = []
    well_keys = []
    well_column_indexes = []
    well_row_indexes = []
    times = []
    temperatures = []
    ODs = []
    

    with pd.ExcelFile(file_path) as excel_file:
        # Loop all the sheets in the file
        for sheet_name in excel_file.sheet_names:
            # Hold all the first OD values with the key being the well name
            initial_ODs = {}

            # Load the current sheet of the excel file
            raw_data_df = pd.read_excel(excel_file, sheet_name)
                        
            # Shared variables each measurement cycle
            cycle_time_in_hours = 0
            cycle_temp = 0

            # Loop all the rows in the dataframe
            for row in raw_data_df.itertuples(index=False):
                # Save the time of the measurement in hours
                if row[0] == "Time [s]":
                    cycle_time_in_hours = (row[1] / 3600)
                # Save the temperature at the time of measurement
                elif row[0] == "Temp. [°C]":
                    cycle_temp = row[1]
                # Save the OD values
                elif row[0] in plate_rows:
                    row_letter = row[0]

                    for current_column in range(0, raw_data_df.shape[1]):
                        if current_column in plate_columns:
                            # if OD is too high it will cause an error when measuring therefore raise an error
                            if row[current_column] == "OVER":
                                err_msg = f'A measurement with the value of "OVER" is in cell {str(((row[0])))}{str(current_column)} at sheet: {sheet_name} in file: {current_file_name}'
                                log.append(err_msg)
                                raise ValueError(err_msg)
                            
                            # Check if a value has alredy been saved for the current well
                            well_loc = f'{row_letter}{current_column}'
                            if not well_loc in initial_ODs:
                                # Save the initial OD value for the current well to later normalize against
                                initial_ODs[well_loc] = row[current_column]
                                
                            # Add data to the lists
                            file_names.append(current_file_name)
                            plate_names.append(sheet_name)
                            # The current_column is given in 1 based indexing, convert to 0 based indexing for consistency
                            well_column_indexes.append(current_column - 1)
                            well_row_indexes.append(gc_utils.convert_row_letter_to_number(row_letter))
                            well_keys.append(well_loc)
                            times.append(cycle_time_in_hours)
                            temperatures.append(cycle_temp)

                            # Normalize the OD by subtracting the initial OD (for the first OD read the the value will be negative therefore set it to 0)
                            ODs.append(row[current_column] - initial_ODs[well_loc] if row[current_column] - initial_ODs[well_loc] > 0 else 0)

    raw_data_df = pd.DataFrame({
        "file_name": file_names, "plate_name": plate_names, "well_key": well_keys , "well_row_index": well_row_indexes,
        "well_column_index": well_column_indexes, "time": times, "temperature": temperatures, "OD": ODs
    })

    raw_data_df = raw_data_df.set_index(["file_name", "plate_name", "well_row_index", "well_column_index"])
    # Sort the index
    return raw_data_df.sort_index()


def save_dataframe_to_csv(df, output_file_path, file_name):
    '''
    Description
    -----------
    Save a dataframe to a csv file with the indexes
    
    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be saved
    output_file_path : str
        The path to the folder where the csv file will be saved
    file_name : str
        The name of the csv file, supply the value of the file name without the extension
    '''
    #Create the output file path with the file name and extension
    file_path_with_file_name = os.path.join(output_file_path, f'{file_name}.csv')
    # Save the dataframe a csv file
    df.to_csv(file_path_with_file_name, index=True)
    return file_path_with_file_name


def create_directory(father_directory, nested_directory_name):
    '''
    Description
    -----------
    Create a directory if it does not exist
    
    Parameters
    ----------
    father_directory : str
        The path to the directory under which the new directory will be created
    nested_directory_name : str
        The name of the nested directory to be created
    '''
    # Create the output directory path
    new_dir_path = os.path.join(father_directory, nested_directory_name)
    # Create the directory if it does not exist
    if not os.path.isdir(new_dir_path):
        os.mkdir(new_dir_path)
    return new_dir_path


def create_single_well_graphs(file_name ,raw_data, summary_data, output_path, title, decimal_percision):
    '''Create graphs from the data collected in previous steps for each well in the experiment
    Parameters
    ----------
    file_name : str
        The name of the file being processed. Will be used to prefix the output file names
    raw_data : pandas.DataFrame
        dataframe returned from the read_tecan_stacker_xlsx function or one with the same structure
    summary_data : pandas.DataFrame
        dataframe returned from the get_experiment_growth_parameters function or one with the same structure
    output_path : str
        Save path
    title: str
        The title for the graphs
    decimal_percision: int
        The amount of digits after the decimal point to show in the labels
    Returns
    -------
    null
    '''

    # Matplotlib backend mode - a non-interactive backend that can only write to files
    # Before changing to this mode the program would crash after the creation of about 250 graphs
    matplotlib.use("Agg")

    plt.style.use('ggplot')

    # Styles
    point_size = 50
    alpha = 0.6

    df_unindexed = raw_data.reset_index()
    # Get all unique keys to loop through
    file_names = df_unindexed['file_name'].unique()
    plate_names = df_unindexed['plate_name'].unique()
    well_row_indexes = df_unindexed['well_row_index'].unique()
    well_column_indexes = df_unindexed['well_column_index'].unique()


    
    for file_name, plate_name, well_row_index, well_column_index in itertools.product(file_names ,plate_names, well_row_indexes, well_column_indexes):
        well_raw_data = raw_data.xs((file_name, plate_name, well_row_index, well_column_index), level=['file_name', 'plate_name', 'well_row_index', 'well_column_index'])
        well_summary_data = (summary_data.xs((file_name, plate_name, well_row_index, well_column_index), level=['file_name', 'plate_name', 'well_row_index', 'well_column_index'])).iloc[0,:]
        

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Time [hours]')
        ax.set_ylabel('OD600')

        ax.plot(well_raw_data["time"], well_raw_data["OD"], color='black')
        
        # If the well is valid graph it with the data from the fitting procedure, otherwise only graph time vs OD as an aid for seeing what went wrong
        if well_summary_data["is_valid"]:
            lag_end_time, lag_end_OD = well_summary_data["lag_end_time"], well_summary_data["lag_end_OD"]
            # lag data
            ax.scatter([lag_end_time], [lag_end_OD], s=point_size ,alpha=alpha, c=['purple'], marker='s',
                        label= f'end of leg phase: {str(round(lag_end_time, decimal_percision))} hours')

            # Max population growth rate
            max_population_gr_time, max_population_gr_OD, max_population_gr_slope = well_summary_data["max_population_gr_time"], well_summary_data["max_population_gr_OD"], well_summary_data["max_population_gr_slope"]
            # Plot the point and the linear function matching the max population growth rate
            ax.axline((max_population_gr_time, max_population_gr_OD), slope=max_population_gr_slope, color='blue', linestyle=':')
            # plot the point on the graph at which this occures
            ax.scatter([max_population_gr_time], [max_population_gr_OD], c=['darkblue'], s=point_size, alpha=alpha, label=f'Min doubling time: {round(well_summary_data["min_doubling_time"], decimal_percision)} hours')

            # End of exponential phase
            exponet_end_time, exponet_end_OD = well_summary_data["exponet_end_time"], well_summary_data["exponet_end_OD"]
            # plot the point with the label
            ax.scatter([exponet_end_time], [exponet_end_OD], c=["brown"], marker='d' ,s=point_size ,alpha=alpha, label=f'95% of growth: {str(round(exponet_end_time, decimal_percision))} hours')

            carrying_capacity = well_summary_data["carrying_capacity"]
            ax.axhline(y=carrying_capacity, color='black', linestyle='dashdot', label=f'Carrying capacity: {(round(carrying_capacity, decimal_percision))}')

            ax.legend(loc="lower right")
        
        # Save the figure
        fig.savefig(os.path.join(output_path, f"well {well_summary_data['well_key']} from {plate_name} in {file_name}.png"))
        plt.close("all")


def create_averaged_replicates_graphs(raw_data_all_replicates, averaged_raw_data, averaged_growth_parameters, output_path, decimal_percision, condition_file_map ,plate_repeats):
    # Use to keep track of if 
    generated_graphes = {}
    
    # Matplotlib backend mode - a non-interactive backend that can only write to files
    # Before changing to this mode the program would crash after the creation of about 250 graphs
    matplotlib.use("Agg")

    plt.style.use('ggplot')
    
    # Grab unique values of each field to iterate over
    well_row_indexes = pd.unique(raw_data_all_replicates.index.get_level_values('well_row_index'))
    well_column_indexes = pd.unique(raw_data_all_replicates.index.get_level_values('well_column_index'))
    file_names = pd.unique(raw_data_all_replicates.index.get_level_values('file_name'))
    plate_names = pd.unique(raw_data_all_replicates.index.get_level_values('plate_name'))

    file_names_plate_names_indexes = list(itertools.product(file_names, plate_names))
    well_indexes = list(itertools.product(well_row_indexes, well_column_indexes))

    all_replicates_combinations = list(itertools.product(file_names_plate_names_indexes, well_indexes))

    for replicate_identifier in all_replicates_combinations:
        curr_file_name = replicate_identifier[0][0]
        curr_plate_name = replicate_identifier[0][1]
        curr_row_index = replicate_identifier[1][0]
        curr_column_index = replicate_identifier[1][1]

        # If the graph for this condition has already been created than continue
        if (curr_file_name, curr_plate_name, curr_row_index, curr_column_index) in generated_graphes:
            continue


        all_files = __find_list_by_value_incondition_file_map(condition_file_map, curr_file_name)
        all_plates = next(sublist for sublist in plate_repeats if curr_plate_name in sublist)

        all_file_and_plates_for_replicate = list(itertools.product(all_files, all_plates))
        # Add the current well index to the the plate file conbinations
        curr_replicate_all_wells = [file_plate_combination + (curr_row_index, curr_column_index) for file_plate_combination
         in all_file_and_plates_for_replicate]

        curr_raw_data = []
        # Get the other replicates from both plate names and file names
        for replicate in curr_replicate_all_wells:
            curr_raw_data.append(raw_data_all_replicates.xs((replicate[0], replicate[1], replicate[2], replicate[3]), level=["file_name", "plate_name", "well_row_index", "well_column_index"]))

        # Grab the unified raw data and the unified summary data using the condition and the plate replica identifier fields
        condition, plate_identifier = curr_raw_data[0].iloc[0].values[4:6]
        curr_well_key = curr_raw_data[0].iloc[0].values[0]

        curr_averaged_raw_data = averaged_raw_data.xs((condition, plate_identifier, curr_well_key), level=["condition", "plate_replica_identifier", "well_key"])
        curr_averaged_summary_data = averaged_growth_parameters.xs((condition, plate_identifier, curr_well_key), level=["condition", "plate_replica_identifier", "well_key"])

        curr_averaged_raw_data = None if isinstance(curr_averaged_raw_data, pd.DataFrame) and curr_averaged_raw_data.empty else curr_averaged_raw_data
        curr_averaged_summary_data = None if isinstance(curr_averaged_summary_data, pd.DataFrame) and curr_averaged_summary_data.empty else curr_averaged_summary_data


        fig, ax = plt.subplots(figsize=(12, 5))
        save_name = __plot_growth_curve_on_ax(ax, decimal_percision, curr_raw_data, curr_averaged_raw_data, curr_averaged_summary_data)

        # Save the figure
        fig.savefig(os.path.join(output_path, f"{save_name}.png"))
        plt.close("all")

        # Figure created successfully, save all the keys to a dictionary to avoid creating graphs multiple times
        generated_graphes.update({tup: 1 for tup in curr_replicate_all_wells})    


def __find_list_by_value_incondition_file_map(condition_file_map, file_name):
    for key, value_list in condition_file_map.items():
        if file_name in value_list:
            return value_list
    return None


def __plot_growth_curve_on_ax(ax, decimal_percision ,raw_data_all_replicates, averaged_raw_data=None, averaged_growth_parameters=None):
    # Make sure that both averaged_growth_parameters and averaged_raw_data are either None or non-None
    is_plot_averaged_data = (averaged_growth_parameters is None and averaged_raw_data is None) or (averaged_growth_parameters is not None and averaged_raw_data is not None)
    assert is_plot_averaged_data, "Both 'averaged_growth_parameters' and 'averaged_raw_data' must either be None or non-None. Either there are no valid replicates for the condition or there are and the two dfs should match"


    size=10
    alpha=0.3
    # Define different markers and colors
    markers = ['o', 's', 'D', 'v', '^', 'p', '*', 'h', 'x', '+']
    
    
    # Iterate over each curve (DataFrame) and plot with different marker and color
    for i, curve in enumerate(raw_data_all_replicates):
        marker = markers[i % len(markers)]
        color = plt.cm.viridis(i / len(raw_data_all_replicates))
        
        plate_name = pd.unique(curve.index.get_level_values('plate_name'))[0]

        ax.scatter(curve['time'], curve['OD'], label=f'{plate_name}', color=color, marker=marker, s=size, alpha=alpha)
    
    if averaged_raw_data is not None:
        ax.plot(averaged_raw_data[('time', 'mean')], averaged_raw_data[('OD', 'median')], label='Median OD line')

    # Use this variable to mark the files of invalid grphas with an 'invalid' prefix
    is_valid_replicate = True
    if averaged_growth_parameters is not None:
        # Lag
        lag_end_time = averaged_growth_parameters[('lag_end_time', 'median')].iloc[0]
        lag_end_OD = averaged_growth_parameters[('lag_end_OD', 'median')].iloc[0]

        ax.scatter(lag_end_time, lag_end_OD, s=size + 45, color='dimgray', marker='*',
                   label=f'Lag end time {lag_end_time:.{decimal_percision}f} hr', zorder=10)
        
        # This should be around (lag_end_time, lag_end_OD) and show the std on the time axis (X)
        lag_end_time_std = averaged_growth_parameters[('lag_end_time', 'std')].iloc[0]
        # This should be around (lag_end_time, lag_end_OD) and show the std on the OD axis (y)
        lag_end_OD_std = averaged_growth_parameters[('lag_end_OD', 'std')]
        
        ax.errorbar(lag_end_time, lag_end_OD, xerr=lag_end_time_std, yerr=lag_end_OD_std, alpha=0.7,
            fmt='none', color='lightgray', ecolor='gray', elinewidth=2, capsize=3)
        
        # Min doubling time
        max_population_gr_time = averaged_growth_parameters[('max_population_gr_time', 'median')].iloc[0]
        max_population_gr_time_std = averaged_growth_parameters[('max_population_gr_time', 'std')].iloc[0]

        max_population_gr_OD = averaged_growth_parameters[('max_population_gr_OD', 'median')].iloc[0]
        max_population_gr_OD_std = averaged_growth_parameters[('max_population_gr_OD', 'std')].iloc[0]

        max_population_gr_slope = averaged_growth_parameters[('max_population_gr_slope', 'median')].iloc[0]

        min_doubling_time = averaged_growth_parameters[('min_doubling_time', 'median')].iloc[0]
        min_doubling_time_std = averaged_growth_parameters[('min_doubling_time', 'std')].iloc[0]
        
        ax.scatter(max_population_gr_time, max_population_gr_OD, s=size + 20, color='dimgray', marker='h', 
                label=f'Min doubling time {min_doubling_time:.{decimal_percision}f} hr/div', zorder=10)

        # Plot error bars for max population growth point
        ax.errorbar(max_population_gr_time, max_population_gr_OD, 
                    xerr=max_population_gr_time_std, yerr=max_population_gr_OD_std, 
                    alpha=0.7, fmt='none', color='lightgray', ecolor='gray', elinewidth=2, capsize=3)

        ax.axline((max_population_gr_time, max_population_gr_OD), slope=max_population_gr_slope, color='red', alpha=alpha, linestyle=':', label='Extrapolated continuation of exp growth')

        # End of exponential phase
        exponet_end_time = averaged_growth_parameters[('exponet_end_time', 'median')].iloc[0]
        exponet_end_time_std = averaged_growth_parameters[('exponet_end_time', 'std')].iloc[0]
        
        exponet_end_OD = averaged_growth_parameters[('exponet_end_OD', 'median')].iloc[0]
        exponet_end_OD_std = averaged_growth_parameters[('exponet_end_OD', 'std')].iloc[0]

        ax.scatter(exponet_end_time, exponet_end_OD, s=size + 20, color='dimgray', marker='^',
                label=f'exponet end time {exponet_end_time:.{decimal_percision}f} hr, OD {exponet_end_OD:.{decimal_percision}f}', zorder=10)

        ax.errorbar(exponet_end_time, exponet_end_OD, xerr=exponet_end_time_std, yerr=exponet_end_OD_std, alpha=0.7,
            fmt='none', color='lightgray', ecolor='gray', elinewidth=2, capsize=3)


        # Carrying_capacity
        carrying_capacity = averaged_growth_parameters[('carrying_capacity', 'median')].iloc[0]
        carrying_capacity_std = averaged_growth_parameters[('carrying_capacity', 'std')].iloc[0]
        ax.axhline(y=carrying_capacity, color='black', linestyle='dashdot', alpha=alpha,
                   label=f'Carrying capacity {carrying_capacity:.{decimal_percision}f}')

    else:
        is_valid_replicate = False

    # Add legend and labels to the plot for better visualization
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OD')
    plt.subplots_adjust(right=0.72)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    status_string = '' if is_valid_replicate else 'invalid_'
    save_name_and_title = f'{status_string}{pd.unique(curve.index.get_level_values("file_name"))[0]} {plate_name.split(".")[0]} {curve.loc[:,"well_key"].iloc[0]}'
    ax.set_title(save_name_and_title)

    return save_name_and_title


def create_replicate_count_heatmap(unified_summary_data, condition_file_map, plate_columns, plate_rows, output_path):

    matplotlib.use("Agg")
    plt.style.use('ggplot')

    condtions = list(condition_file_map.keys())
    plates = pd.unique(unified_summary_data.index.get_level_values('plate_replica_identifier'))

    condition_plate_combinations = list(itertools.product(condtions, plates))

    heatmap_columns = sorted(plate_columns)
    heatmap_rows = sorted(plate_rows)

    heatmap_rows_index = {char: idx for idx, char in enumerate(heatmap_rows)}
    # The rows don't require anything so fancy since it's 1 based counting for the final index in the matrix

    # Use to generate summary heatmaps
    count_matrices_by_condition = {}
    count_matrices_by_plate = {}
    


    # Per plate per condition heatmap
    for condition_plate_combination in condition_plate_combinations:
        # This way for wells that were completly invalid in all repeats there would still be a 0 as the count
        curr_plate_counts_matrix = np.zeros((len(heatmap_rows), len(heatmap_columns)))

        curr_rep_data_df = unified_summary_data.xs((condition_plate_combination[0], condition_plate_combination[1]),
                                                   level=['condition', 'plate_replica_identifier'])
        
        for well_data in curr_rep_data_df.iterrows():
            well_key = well_data[0]
            well_row = heatmap_rows_index[well_key[0]]
            well_col = int(well_key[1:]) - 1
            # All the fields have the count
            well_rep_count = well_data[1]['lag_end_time']['count']

            curr_plate_counts_matrix[well_row, well_col] = well_rep_count

        # Save the count matrix under the condition and under the plate to later show a summary of them
        condition = condition_plate_combination[0]
        count_matrices_by_condition.setdefault(condition, [curr_plate_counts_matrix]).append(curr_plate_counts_matrix)
        
        plate = condition_plate_combination[1]
        count_matrices_by_plate.setdefault(plate, [curr_plate_counts_matrix]).append(curr_plate_counts_matrix)

        __plot_heatmap(curr_plate_counts_matrix, heatmap_rows, heatmap_columns, condition, plate, output_path)

    # Per condition summary
    for key, count_matrices in count_matrices_by_condition.items():
        condition_count_matrix = np.minimum.reduce(count_matrices)
        __plot_heatmap(condition_count_matrix, heatmap_rows, heatmap_columns, key, 'All plates', output_path)

    # Per plate summay
    for key, count_matrices in count_matrices_by_plate.items():
        condition_count_matrix = np.minimum.reduce(count_matrices)
        __plot_heatmap(condition_count_matrix, heatmap_rows, heatmap_columns, 'All condiotions', key, output_path)


def __plot_heatmap(counts_matrix, heatmap_rows, heatmap_columns, condition, plate, output_path):
    plt.figure(figsize=(10, 6))
    sns.heatmap(counts_matrix, annot=True, cmap="vlag", cbar=False,
                xticklabels=heatmap_columns, yticklabels=heatmap_rows)
    
    # Set the position of column labels to the top
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.yticks(rotation=0)

    title_and_save_name = f'{condition}, {plate}'
    plt.title(title_and_save_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, title_and_save_name))
    plt.close('all')


def create_correlation_panel(unified_summary_data, output_path):
    medians_df = unified_summary_data.xs('median', axis=1, level=1)

    medians_df['exponet_length_in_time'] = medians_df['exponet_end_time'] - medians_df['lag_end_time']
    medians_df['exponet_length_in_OD'] = medians_df['exponet_end_OD'] - medians_df['lag_end_OD']

    corr_matrix = medians_df.corr(method='pearson')


    #mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(15, 15))
    #sns.heatmap(corr_matrix, mask=mask, annot=True, cmap="vlag", vmin=-1, vmax=1, center=0)
    sns.heatmap(corr_matrix, annot=True, cmap="vlag", vmin=-1, vmax=1, center=0)
    heatmap_file = os.path.join(output_path, 'pearson_correlation_heatmap.png')
    plt.title('Pearson Correlation Heatmap between all feature pairs', fontsize=24)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(heatmap_file)
    plt.close()

    columns = medians_df.columns
    for i in range(len(columns)):
        for j in range(i+1, len(columns)):
            col_x = columns[i]
            col_y = columns[j]

            plt.figure(figsize=(7, 7))
            sns.scatterplot(data=medians_df, x=col_x, y=col_y)
            slope, intercept = np.polyfit(medians_df[col_x], medians_df[col_y], 1)
            plt.plot(medians_df[col_x], slope * medians_df[col_x] + intercept, color='red', linestyle='--')

            pearson_corr, p_value = pearsonr(medians_df[col_x], medians_df[col_y])
            plt.title(f'{col_x} vs {col_y} (Pearson r: {pearson_corr:.2f}), p value: {p_value:.4f}')

            scatter_file = os.path.join(output_path, f'scatter_{col_x}_vs_{col_y}.png')
            plt.tight_layout()
            plt.savefig(scatter_file)
            plt.close()

    print(f"Heatmap saved to {heatmap_file}")
    print(f"Scatter plots saved in {output_path}")


def plot_dist(relative_CC_scores):
# Set the figure size
    plt.figure(figsize=(10, 10))
    
    # Plot the histogram using seaborn, without KDE for simplicity
    hist_data = sns.histplot(relative_CC_scores, bins=20, kde=True, color="royalblue")
    
    # Calculate the total number of samples
    total_samples = len(relative_CC_scores)
    
    # Add percentage labels for each bar
    for patch in hist_data.patches:
        # Get the height of the current bar (number of samples in the bin)
        height = patch.get_height()
        
        # Calculate the percentage
        percent = (height / total_samples) * 100
        
        # Place the text label
        plt.text(patch.get_x() + patch.get_width() / 2, height + 0.5, 
                 f'{percent:.1f}%', ha='center', fontsize=10, color='black')
    
    # Add title and labels
    plt.title("Distribution of Relative CC Scores", fontsize=16)
    plt.xlabel("Relative CC Scores", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    
    # Show grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Show the plot
    plt.show()


def import_previous_run_data(output_path):
    '''
    Desrciption
    -----------
    Read the content the output file with the results of a previous run.
    The folder must include the multiple repeat comprison df with it's originial name,
    At least one raw data df (as saved from the read_tecan_stacker_xlsx or any future version of the function that outputs the same file structure)
    and the same number of summary files belonging to the same original file. If any of these assumptions is not true the function will raise a value error.
    
    Parameters
    ----------
    output_path : str
        The path to the output file with the results of a previous run
    
    Returns
    -------
    file_raw_data_df_mapping : dictionary
        The name of the file as the key and the raw data from the file as a pandas.DataFrame as the value
    file_summary_df_mapping : dictionary
        The name of the file as the key and the summary data from get_experiment_growth_parameters as a pandas.DataFrame as the value
    variation_matrix : pandas.DataFrame
        The muliple repeat comprison table as returned from get_reps_variation_data
    '''
    
    # Initialize the dictionaries for raw data and summary data
    file_raw_data_df_mapping = {}
    file_summary_df_mapping = {}
    variation_matrix = None
    
    # List all files in the specified directory
    all_files = os.listdir(output_path)
    
    # Filter files based on the required suffix
    raw_data_files = [f for f in all_files if f.endswith('_raw_data.csv')]
    summary_data_files = [f for f in all_files if f.endswith('_summary_data.csv')]
    variation_data_files = [f for f in all_files if f.endswith('variation_matrix.csv')]
    
    # Check that the number of raw data files matches the number of summary data files
    if len(raw_data_files) != len(summary_data_files):
        raise ValueError("The number of raw data files does not match the number of summary data files.")
    
    # Check that there is exactly one variation matrix file
    if len(variation_data_files) != 1:
        raise ValueError("There must be exactly one variation_matrix file.")
    
    # Process raw data files
    for raw_file in raw_data_files:
        file_base_name = raw_file.replace('_raw_data.csv', '')
        raw_data_path = os.path.join(output_path, raw_file)
        # Index the df the same way it was indexed intially
        file_raw_data_df_mapping[file_base_name] = pd.read_csv(raw_data_path).set_index(["file_name", "plate_name", "well_row_index", "well_column_index"])
    
    # Process summary data files
    for summary_file in summary_data_files:
        file_base_name = summary_file.replace('_summary_data.csv', '')
        summary_data_path = os.path.join(output_path, summary_file)
        file_summary_df_mapping[file_base_name] = pd.read_csv(summary_data_path).set_index(["file_name", "plate_name", "well_row_index", "well_column_index"])
    
    # Load the variation matrix
    variation_matrix_path = os.path.join(output_path, variation_data_files[0])
    variation_matrix = pd.read_csv(variation_matrix_path).set_index(['file_name_A', 'file_name_B', 'plate_name_A', 'plate_name_B', 'well_row_index', 'well_column_index'])
    


    return file_raw_data_df_mapping, file_summary_df_mapping, variation_matrix