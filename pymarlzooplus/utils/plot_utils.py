import json
import os
import random
import pickle
import argparse
import sys
import yaml
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


PREDEFINED_MAP_ALGO_COLORS = {
    'QMIX': '#4169E1',  # Royal blue
    'QPLEX': '#32CD32',  # Lime green
    'MAA2C': '#FF6347',  # Tomato
    'MAPPO': '#40E0D0',  # Turquoise
    'HAPPO': '#AFEEEE',  # Pale turquoise (more vibrant than pastel green)
    'MAT-DEC': '#000000',  # Black
    'COMA': '#9370DB',  # Medium purple (richer than plum)
    'EOI': '#FFD700',  # Gold (bright alternative to yellow)
    'MASER': '#C71585',  # Medium violet red (stronger than light red)
    'EMC': '#A9A9A9',  # Dark gray (more visible than light gray)
    'CDS': '#964B00',  # Brown
    'MAVEN': '#FF69B4',  # Pink
    'CommFormer': '#008000',  # Green
    'MAGIC': '#00A6D6',  # Cyan
    'IQL': '#FF8C00',  # Dark orange

}

def _plot_custom_legend():
    """
    Helper function to create and plot the predefined horizontal MARL legend
    below the current figure.
    """
    patches = [
        plt.Line2D([0], [0], color=color, marker='o', markersize=10, label=algo, linestyle='None',
                   markeredgewidth=1.5)
        for algo, color in PREDEFINED_MAP_ALGO_COLORS.items()
    ]
    plt.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=6, frameon=False, fontsize='small', handletextpad=0.5, columnspacing=1)


def base_read_json(json_path):
    """
    json_path: The path to a .json file
    """

    assert os.path.exists(json_path), \
        f"The provided path to json file does not exist! \n'json_path': {json_path}"

    # Open the file for reading
    with open(json_path, 'r') as file:
        # Load data from the file into a Python dictionary
        data = json.load(file)

    return data


def read_json(json_path):
    """
        json_path: The path to info.json file
        """

    assert os.path.basename(json_path) == 'info.json', \
        f"The provided path {json_path} is not a path of a info.json file!"

    try:
        data = base_read_json(json_path)
    except:
        # In case that it fails to load info.json, try metrics.json
        json_base_path = os.path.dirname(json_path)
        json_metrics_path = os.path.join(json_base_path, "metrics.json")
        data = base_read_json(json_metrics_path)
        # Transform data in the form of info.json
        new_data = {}
        for data_key in data.keys():
            data_values = data[data_key]['values']
            data_t = data[data_key]['steps']
            new_data[data_key] = data_values
            new_data[data_key + '_T'] = data_t
        data = new_data

    return data


def create_plot(
        x_data,
        y_data_mean,
        y_data_std,
        path_to_save,
        x_label,
        y_label,
        plot_title,
        legend_labels=None
):

    assert len(x_data) == len(y_data_mean), \
        f"'len(x_data)': {len(x_data)}, 'len(y_data_mean)': {len(y_data_mean)}"
    if y_data_std[0] is not None:
        assert len(y_data_std) == len(y_data_mean), \
            f"'len(y_data_std)': {len(y_data_std)}, 'len(y_data_mean)': {len(y_data_mean)}"

    # Create a new figure
    plt.figure()

    for data_idx in range(len(x_data)):

        # Plot the data
        plt.plot(
            x_data[data_idx],
            y_data_mean[data_idx],
            label=None if legend_labels is None else legend_labels[data_idx],
        )

        # Add std if available
        if y_data_std[0] is not None:
            # Calculate the upper and lower bounds of the standard deviation
            std_upper = np.array(y_data_mean[data_idx]) + 1.15*np.array(y_data_std[data_idx])  # 75%
            std_lower = np.array(y_data_mean[data_idx]) - 1.15*np.array(y_data_std[data_idx])  # 75%
            # Add a shaded area for the standard deviation
            plt.fill_between(x_data[data_idx], std_lower, std_upper, alpha=0.2)

    if legend_labels is not None:
        # Adding legend
        plt.legend()

    # Add labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.tight_layout()

    # Save and close
    plt.savefig(path_to_save)
    plt.close()


def _extract_metric_values(metric_data):
    if not metric_data:
        return None
    values = []
    dtype = None
    for item in metric_data:
        if isinstance(item, dict):
            if 'value' not in item:  # No values recorded, skip this metric
                return None
            dtype = dtype or item.get('dtype', None)
            values.append(item['value'])
        else:
            values.append(item)
    return np.array(values, dtype=dtype)


def get_mean_and_std_data(results_data, results_type):

    # Mean values
    x_data = results_data[results_type + "_T"]
    mean_data = _extract_metric_values(results_data[results_type])
    if mean_data is None:
        return None, None, None

    # Std values
    std_data_key = "_".join(results_type.split("_")[:-1]) + "_std"
    std_data = None if std_data_key not in results_data.keys() else _extract_metric_values(results_data[std_data_key])
    if std_data is not None and len(std_data) != len(mean_data):
        std_data = None

    return x_data, mean_data, std_data


def get_return_data(results_data):

    # Get "return" data for x and y axes
    x_return_data, return_mean_data, return_std_data = \
        get_mean_and_std_data(results_data, "return_mean")
    # Get "test_return" data for x and y axes
    x_test_return_data, test_return_mean_data, test_return_std_data = \
        get_mean_and_std_data(results_data, "test_return_mean")

    # Get "ep_length_mean" to divide "return_mean_data"
    _, ep_length_mean_data, _ = \
        get_mean_and_std_data(results_data, "ep_length_mean")
    # Get "test_ep_length_mean" to divide "test_return_mean_data"
    _, test_ep_length_mean_data, _ = \
        get_mean_and_std_data(results_data, "test_ep_length_mean")

    # Calculate the normalized returns
    assert len(return_mean_data) == len(ep_length_mean_data), \
        f"'len(return_mean_data)': {len(return_mean_data)}, " + \
        f"'len(ep_length_mean_data)': {len(ep_length_mean_data)}"
    normalized_return_mean_data = np.array(return_mean_data) / np.array(ep_length_mean_data)
    normalized_return_std_data = np.array(return_std_data) / np.array(ep_length_mean_data)
    assert len(test_return_mean_data) == len(test_ep_length_mean_data), \
        f"'len(test_return_mean_data)': {len(test_return_mean_data)}, " + \
        f"'len(test_ep_length_mean_data)': {len(test_ep_length_mean_data)}"
    test_normalized_return_mean_data = np.array(test_return_mean_data) / np.array(test_ep_length_mean_data)
    test_normalized_return_std_data = np.array(test_return_std_data) / np.array(test_ep_length_mean_data)

    return (
        x_return_data,
        return_mean_data,
        return_std_data,
        x_test_return_data,
        test_return_mean_data,
        test_return_std_data,
        normalized_return_mean_data,
        normalized_return_std_data,
        test_normalized_return_mean_data,
        test_normalized_return_std_data
    )


def plot_single_experiment_results(path_to_results, algo_name, env_name):
    """
    path_to_results: str, a single path where inside there is the "info.json" file.
    algo_name: str, name of the algorithm, e.g., "qmix"
    env_name: str, name of the environment, e.g., "rware_v1:rware_v1-tiny-2ag-v1"
    """

    assert os.path.exists(path_to_results), \
        f"The provided path to results does not exist! \n'path_to_results': {path_to_results}"

    # Get results
    path_to_info_json = os.path.join(path_to_results, 'info.json')
    results_data = read_json(path_to_info_json)

    plot_title = "Algo: " + algo_name + ", Env: " + env_name
    path_to_save_results = os.path.join(path_to_results, "plots")
    if not os.path.exists(path_to_save_results):
        os.mkdir(path_to_save_results)

    for results_type in results_data.keys():

        # Ignore "timesteps" and "std".
        # These are used only in combination with mean metric values.
        # Also, completely ignore "episode", and "test_return" since it will be plotted with "return".
        if "_T" in results_type or \
           "_std" in results_type or \
           "episode" in results_type or \
           "test_return" in results_type:
            continue

        if "return" in results_type:

            # Get "returns" and normalized "returns" data
            (
                x_return_data,
                return_mean_data,
                return_std_data,
                x_test_return_data,
                test_return_mean_data,
                test_return_std_data,
                normalized_return_mean_data,
                normalized_return_std_data,
                test_normalized_return_mean_data,
                test_normalized_return_std_data
             ) = get_return_data(results_data)

            # Create the plot for the unnormalized returns
            path_to_save = os.path.join(path_to_save_results, "return_mean")
            create_plot(
                [x_return_data, x_test_return_data],
                [return_mean_data, test_return_mean_data],
                [return_std_data, test_return_std_data],
                path_to_save,
                "Steps",
                "Episodic Reward",
                plot_title,
                legend_labels=["Train", "Test"]
            )

            # Create the plot for the normalized returns
            path_to_save = os.path.join(path_to_save_results, "normalized_return_mean")
            create_plot(
                [x_return_data, x_test_return_data],
                [normalized_return_mean_data, test_normalized_return_mean_data],
                [normalized_return_std_data, test_normalized_return_std_data],
                path_to_save,
                "Steps",
                "Per-Step Episodic Reward",
                plot_title,
                legend_labels=["Train", "Test"]
            )

        else:
            # Get data for x and y axes
            x_data, mean_data, std_data = get_mean_and_std_data(results_data, results_type)
            if mean_data is None:  # No values found, skip it
                continue
            # Define y_label
            results_type_for_y_axis_label = " ".join([elem for elem in results_type.split("_") if elem != "mean"])
            # Define where to save the plot
            path_to_save = os.path.join(path_to_save_results, results_type)
            # Create the plot
            create_plot(
                [x_data],
                [mean_data],
                [std_data],
                path_to_save,
                "Steps",
                results_type_for_y_axis_label,
                plot_title
            )


def calculate_mean_and_std_of_multiple_exps(x_data_list, mean_data_list):

    def truncate_data(x, y, max_time):
        """ Truncate data to minimum max_time """
        valid_indices = x <= max_time
        return x[valid_indices], y[valid_indices]

    # Step 1: Find the maximum common ending timestep
    max_common_time = min([max(x) for x in x_data_list])

    # Step 2: Truncate all datasets
    truncated_data = [
        truncate_data(np.array(x), np.array(y), max_common_time) for x, y in zip(x_data_list, mean_data_list)
    ]

    # Step 3: Define a common set of timesteps and interpolate
    # Increase the data resolution by a factor of 10.
    common_timeline = np.linspace(1, max_common_time, num=len(truncated_data[0][0])*10)
    interpolated_data = np.array([np.interp(common_timeline, x, y) for x, y in truncated_data])

    # Step 4: Calculate mean and standard deviation
    mean_data = np.mean(interpolated_data, axis=0)
    std_data = np.std(interpolated_data, axis=0)

    n_samples = interpolated_data.shape[0]

    return mean_data, std_data, common_timeline, n_samples


def create_multiple_exps_plot(
        all_results,
        path_to_save,
        plot_title,
        legend_labels,  # Algorithm names
        env_name,
        plot_train=True,
        plot_legend_bool=False
):
    # Font size configuration
    title_fontsize = 18
    label_fontsize = 16

    # Set global parameters for font sizes
    mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick label font size
    mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick label font size

    # Create new figures, one for returns and another one for per-set returns.
    plt.figure(1)
    plt.xlabel("Steps", fontsize=label_fontsize)
    plt.ylabel("Episodic Reward", fontsize=label_fontsize)
    plt.title(plot_title, fontsize=title_fontsize)

    plt.figure(2)
    plt.xlabel("Steps", fontsize=label_fontsize)
    plt.ylabel("Per-step Episodic Reward", fontsize=label_fontsize)
    plt.title(plot_title, fontsize=title_fontsize)

    lines = []  # To keep track of plot lines for legend
    plot_legends = []  # To keep track of plot legends
    extra_lines = []  # To track lines not in PREDEFINED_MAP_ALGO_COLORS
    extra_plot_legends = []  # To track labels not in PREDEFINED_MAP_ALGO_COLORS

    for alg_idx in range(len(all_results)):

        mean_data = all_results[alg_idx][0]
        std_data = all_results[alg_idx][1]
        norm_mean_data = all_results[alg_idx][2]
        norm_std_data = all_results[alg_idx][3]
        common_timeline = all_results[alg_idx][4]
        test_mean_data = all_results[alg_idx][5]
        test_std_data = all_results[alg_idx][6]
        norm_test_mean_data = all_results[alg_idx][7]
        norm_test_std_data = all_results[alg_idx][8]
        test_common_timeline = all_results[alg_idx][9]
        n_samples = all_results[alg_idx][10]

        # Check data consistency
        assert (len(mean_data) ==
                len(norm_mean_data) ==
                len(common_timeline)), \
            (f"'len(mean_data)': {len(mean_data)}, "
             f"\n'len(norm_mean_data)': {len(norm_mean_data)}, "
             f"\n'len(common_timeline)': {len(common_timeline)}, ")
        assert (len(test_mean_data) ==
                len(norm_test_mean_data) ==
                len(test_common_timeline)), \
            (f"'len(test_mean_data)': {len(test_mean_data)}, "
             f"\n'len(norm_test_mean_data)': {len(norm_test_mean_data)}, "
             f"\n'len(test_common_timeline)': {len(test_common_timeline)}, ")
        if std_data is not None:
            assert (len(std_data) ==
                    len(norm_std_data) ==
                    len(common_timeline)), \
                (f"'len(std_data)': {len(std_data)}, "
                 f"'len(norm_std_data)': {len(norm_std_data)}, "
                 f"'len(common_timeline)': {len(common_timeline)}")
            assert (len(test_std_data) ==
                    len(norm_test_std_data) ==
                    len(test_common_timeline)), \
                (f"'len(test_std_data)': {len(test_std_data)}, "
                 f"'len(norm_test_std_data)': {len(norm_test_std_data)}, "
                 f"'len(test_common_timeline)': {len(test_common_timeline)}")

        data_for_plots = [
            [
                mean_data,
                std_data,
                common_timeline,
                test_mean_data,
                test_std_data,
                test_common_timeline
            ],
            [
                norm_mean_data,
                norm_std_data,
                common_timeline,
                norm_test_mean_data,
                norm_test_std_data,
                test_common_timeline
            ]
        ]

        # Define the label to plot in legend
        plot_legend = legend_labels[alg_idx] if plot_train is False else legend_labels[alg_idx] + "-test"

        # Retrieve color based on 'plot_legend', or generate a random color if label is not in color_map
        color = PREDEFINED_MAP_ALGO_COLORS.get(plot_legend, f'#{random.randint(0, 0xFFFFFF):06x}')

        # Keep the used plot_legend
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            plot_legends.append(plot_legend)
        else:
            extra_plot_legends.append(plot_legend)

        for data_for_plot_idx, data_for_plot in enumerate(data_for_plots):

            # Set which figure to update
            plt.figure(data_for_plot_idx+1)

            # Plot the test data
            line, = plt.plot(
                data_for_plot[5],
                data_for_plot[3],
                label=plot_legend,
                color=color
            )

            # Append to either main or extra lines list
            if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
                if data_for_plot_idx == 0:
                    lines.append([])
                lines[-1].append([line])
            else:
                if data_for_plot_idx == 0:
                    extra_lines.append([])
                extra_lines[-1].append([line])

            # Add std if available
            if data_for_plot[4] is not None:

                # Calculate the upper and lower bounds of the standard deviation
                std_upper = np.array(data_for_plot[3]) + 1.15*np.array(data_for_plot[4]) / np.sqrt(n_samples)  # 75%
                std_lower = np.array(data_for_plot[3]) - 1.15*np.array(data_for_plot[4]) / np.sqrt(n_samples)  # 75%

                # Add a shaded area for the standard deviation
                plt.fill_between(data_for_plot[5], std_lower, std_upper, alpha=0.2, color=color)

            # Plot the train data
            if plot_train is True:
                train_plot_legend = legend_labels[alg_idx] + "-train"
                train_color = f'#{random.randint(0, 0xFFFFFF):06x}'
                line, = plt.plot(
                    data_for_plot[2],
                    data_for_plot[0],
                    label=train_plot_legend,
                    color=train_color
                )

                # Append to either main or extra lines list
                # based on the 'plot_legend' since 'train_plot_legend'
                # is not in 'PREDEFINED_MAP_ALGO_COLORS' by default
                if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
                    lines[-1][-1].append(line)
                else:
                    extra_lines[-1][-1].append(line)

                # Add std if available
                if data_for_plot[1] is not None:
                    # Calculate the upper and lower bounds of the standard deviation
                    std_upper = np.array(data_for_plot[0]) + np.array(data_for_plot[1])
                    std_lower = np.array(data_for_plot[0]) - np.array(data_for_plot[1])
                    # Add a shaded area for the standard deviation
                    plt.fill_between(data_for_plot[2], std_lower, std_upper, alpha=0.2, color=train_color)

    ## Custom legend creation
    legend_order = list(PREDEFINED_MAP_ALGO_COLORS.keys())
    legend_lines_fig_1 = []
    legend_lines_fig_2 = []
    # At first, it should be the lines which are listed in 'legend_order'
    for __plot_legend in legend_order:
        if __plot_legend in plot_legends:
            # Find the index of '__plot_legend' in 'plot_legends'
            _plot_legend_idx = plot_legends.index(__plot_legend)
            # Add the lines for figure 1
            legend_lines_fig_1.extend(lines[_plot_legend_idx][0])
            # Add the lines for figure 2
            legend_lines_fig_2.extend(lines[_plot_legend_idx][1])
    # Then should be all the rest lines (of algorithms not listed in 'legend_order')
    for _lines in extra_lines:
        # Add the lines for figure 1
        legend_lines_fig_1.extend(_lines[0])
        # Add the lines for figure 2
        legend_lines_fig_2.extend(_lines[1])

    # Adding legend, save, and close
    plt.figure(1)
    plt.tight_layout()
    if plot_legend_bool:
        _plot_custom_legend()

    path_to_save_plot = os.path.join(path_to_save, f"return_mean_env={env_name}")
    plt.savefig(path_to_save_plot, bbox_inches='tight')
    plt.close()

    plt.figure(2)
    plt.tight_layout()
    if plot_legend_bool:
        _plot_custom_legend()

    path_to_save_plot = os.path.join(path_to_save, f"normalized_return_mean_env={env_name}")
    plt.savefig(path_to_save_plot, bbox_inches='tight')
    plt.close()

    # Resetting the following parameters to their default values
    mpl.rcParams['xtick.labelsize'] = mpl.rcParamsDefault['xtick.labelsize']
    mpl.rcParams['ytick.labelsize'] = mpl.rcParamsDefault['ytick.labelsize']


def calculate_mean_best_reward_over_multiple_experiments(
        all_results,
        path_to_save,
        algo_names,
        env_name,
        n_last_values=50
):

    def truncate_data(y, max_time):
        """ Truncate data to minimum max_time """
        y = np.array([y_element[:max_time] for y_element in y])
        return y

    def get_max_indices(_data):
        all_values_best_idx = np.argmax(_data, axis=1)
        last_values_best_idx = np.argmax(np.array(_data)[:, -n_last_values:], axis=1)
        return all_values_best_idx, last_values_best_idx

    def get_best_rewards(_data, all_values_best_idx, last_values_best_idx):

        _best_rewards = {}

        row_indices = np.arange(_data.shape[0])
        _best_rewards['overall_mean_max_reward'] = np.mean(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_std_max_reward'] = np.std(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_median_max_reward'] = np.median(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_25_percentile_max_reward'] = np.percentile(_data[row_indices, all_values_best_idx], 25)
        _best_rewards['overall_75_percentile_max_reward'] = np.percentile(_data[row_indices, all_values_best_idx], 75)
        _best_rewards['overall_min_max_reward'] = np.min(_data[row_indices, all_values_best_idx])
        _best_rewards['overall_max_max_reward'] = np.max(_data[row_indices, all_values_best_idx])

        last_values_data = _data[:, -n_last_values:]
        _best_rewards['last_values_mean_max_reward'] = np.mean(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_std_max_reward'] = np.std(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_median_max_reward'] = np.median(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_25_percentile_max_reward'] = \
            np.percentile(last_values_data[row_indices, last_values_best_idx], 25)
        _best_rewards['last_values_75_percentile_max_reward'] = \
            np.percentile(last_values_data[row_indices, last_values_best_idx], 75)
        _best_rewards['last_values_min_max_reward'] = np.min(last_values_data[row_indices, last_values_best_idx])
        _best_rewards['last_values_max_max_reward'] = np.max(last_values_data[row_indices, last_values_best_idx])

        # Round the results
        for key, value in _best_rewards.items():
            _best_rewards[key] = round(value, 2)

        return _best_rewards

    # Prepare the csv columns and data. These metrics have the same order as the 'best_rewards'
    metrics = [
        'test_overall_mean_max_reward',
        'test_overall_std_max_reward',
        'test_overall_median_max_reward',
        'test_overall_25_percentile_max_reward',
        'test_overall_75_percentile_max_reward',
        'test_overall_min_max_reward',
        'test_overall_max_max_reward',
        'test_last_values_mean_max_reward',
        'test_last_values_std_max_reward',
        'test_last_values_median_max_reward',
        'test_last_values_25_percentile_max_reward',
        'test_last_values_75_percentile_max_reward',
        'test_last_values_min_max_reward',
        'test_last_values_max_max_reward',
        'train_overall_mean_max_reward',
        'train_overall_std_max_reward',
        'train_overall_median_max_reward',
        'train_overall_25_percentile_max_reward',
        'train_overall_75_percentile_max_reward',
        'train_overall_min_max_reward',
        'train_overall_max_max_reward',
        'train_last_values_mean_max_reward',
        'train_last_values_std_max_reward',
        'train_last_values_median_max_reward',
        'train_last_values_25_percentile_max_reward',
        'train_last_values_75_percentile_max_reward',
        'train_last_values_min_max_reward',
        'train_last_values_max_max_reward',
        'test_norm_overall_mean_max_reward',
        'test_norm_overall_std_max_reward',
        'test_norm_overall_median_max_reward',
        'test_norm_overall_25_percentile_max_reward',
        'test_norm_overall_75_percentile_max_reward',
        'test_norm_overall_min_max_reward',
        'test_norm_overall_max_max_reward',
        'test_norm_last_values_mean_max_reward',
        'test_norm_last_values_std_max_reward',
        'test_norm_last_values_median_max_reward',
        'test_norm_last_values_25_percentile_max_reward',
        'test_norm_last_values_75_percentile_max_reward',
        'test_norm_last_values_min_max_reward',
        'test_norm_last_values_max_max_reward',
        'train_norm_overall_mean_max_reward',
        'train_norm_overall_std_max_reward',
        'train_norm_overall_median_max_reward',
        'train_norm_overall_25_percentile_max_reward',
        'train_norm_overall_75_percentile_max_reward',
        'train_norm_overall_min_max_reward',
        'train_norm_overall_max_max_reward',
        'train_norm_last_values_mean_max_reward',
        'train_norm_last_values_std_max_reward',
        'train_norm_last_values_median_max_reward',
        'train_norm_last_values_25_percentile_max_reward',
        'train_norm_last_values_75_percentile_max_reward',
        'train_norm_last_values_min_max_reward',
        'train_norm_last_values_max_max_reward',
    ]
    # Create an empty DataFrame with metrics as the index
    df = pd.DataFrame(index=metrics, columns=algo_names)

    for alg_idx in range(len(all_results)):

        mean_data = all_results[alg_idx][11]
        norm_mean_data = all_results[alg_idx][12]
        test_mean_data = all_results[alg_idx][13]
        norm_test_mean_data = all_results[alg_idx][14]

        # Step 1: Find the maximum common ending timestep
        max_common_time = min([len(x) for x in mean_data])
        test_max_common_time = min([len(x) for x in test_mean_data])

        # Step 2: Truncate all datasets
        data_for_reward_calculation = [
            truncate_data(test_mean_data, test_max_common_time),
            truncate_data(mean_data, max_common_time),
            truncate_data(norm_test_mean_data, test_max_common_time),
            truncate_data(norm_mean_data, max_common_time)
        ]

        data_columns = [
            'test',
            'train',
            'norm test',
            'norm train'
        ]

        test_all_values_best_idx = None
        test_last_values_best_idx = None
        train_all_values_best_idx = None
        train_last_values_best_idx = None
        algo_data_for_csv = np.zeros((len(metrics),), dtype=np.float32)
        for data_idx, (data, data_column) in enumerate(zip(data_for_reward_calculation, data_columns)):

            # Get indices from all experiments of the best reward over the last "n_last_values" and over all values
            if data_column == 'test':
                test_all_values_best_idx, test_last_values_best_idx = get_max_indices(data)
            elif data_column == 'train':
                train_all_values_best_idx, train_last_values_best_idx = get_max_indices(data)

            # Get the best reward statistics
            best_rewards = None
            if 'test' in data_column:
                best_rewards = get_best_rewards(data, test_all_values_best_idx, test_last_values_best_idx)
            elif 'train' in data_column:
                best_rewards = get_best_rewards(data, train_all_values_best_idx, train_last_values_best_idx)
            else:
                raise ValueError(f'data_column: {data_column}')

            # Assign the statistics to the csv data
            algo_data_for_csv[(data_idx * len(best_rewards)): ((data_idx + 1) * len(best_rewards))] = \
                list(best_rewards.values())

        # Assign the algo values to the dataframe
        df[algo_names[alg_idx]] = algo_data_for_csv.copy()

    # Save dataframe
    file_path = os.path.join(path_to_save, f'best_rewards_env={env_name}.csv')
    df.to_csv(file_path)


def plot_multiple_experiment_results(
        paths_to_results,
        algo_names,
        env_name,
        path_to_save,
        plot_train,
        plot_legend_bool
):
    """
    path_to_results: list of str, all paths of the algorithms results for a specific environment.
                     Each path should contain folders like: 1, 2, 3, e.t.c., where each one should have inside
                     a file "info.json". NOTE: The order of paths should be aligned with the other of "algo_names" list.
    algo_names: list of str, all the algorithm names, e.g., ["qmix", "qplex", "maa2c", ...]
    env_name: str, name of the environment, e.g., "rware_v1:rware_v1-tiny-2ag-v1" .
    path_to_save: str, path to save the plots.
    plot_train: bool, whether to plot the training returns or not.
    """

    assert len(paths_to_results) == len(algo_names), \
        f"'len(paths_to_results)': {len(paths_to_results)}, \n'len(algo_names)': {len(algo_names)}"

    all_results = []
    for path_to_results_idx, path_to_results in enumerate(paths_to_results):

        # Check if the provided path is valid
        assert os.path.exists(path_to_results), \
            f"The provided 'path_to_results' does not exist! 'path_to_results': {path_to_results}"

        path_to_exps = [
            os.path.join(path_to_results, elem) for elem in os.listdir(path_to_results) if elem.isdigit()
        ]
        x_return_data_list = []
        return_mean_data_list = []
        x_test_return_data_list = []
        test_return_mean_data_list = []
        normalized_return_mean_data_list = []
        test_normalized_return_mean_data_list = []
        for path_to_exp in path_to_exps:

            # Get results
            path_to_info_json = os.path.join(path_to_exp, 'info.json')
            results_data = read_json(path_to_info_json)

            # Get "returns" and normalized "returns" data
            (
                x_return_data,
                return_mean_data,
                _,
                x_test_return_data,
                test_return_mean_data,
                _,
                normalized_return_mean_data,
                _,
                test_normalized_return_mean_data,
                _
            ) = get_return_data(results_data)

            # Keep returns to compute their mean and std
            x_return_data_list.append(x_return_data)
            return_mean_data_list.append(return_mean_data)
            x_test_return_data_list.append(x_test_return_data)
            test_return_mean_data_list.append(test_return_mean_data)
            normalized_return_mean_data_list.append(normalized_return_mean_data)
            test_normalized_return_mean_data_list.append(test_normalized_return_mean_data)

        (
            mean_data,
            std_data,
            common_timeline,
            n_samples
        ) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, return_mean_data_list)
        (
            norm_mean_data,
            norm_std_data,
            _,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_return_data_list, normalized_return_mean_data_list)
        (
            test_mean_data,
            test_std_data,
            test_common_timeline,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_return_mean_data_list)
        (
            norm_test_mean_data,
            norm_test_std_data,
            _,
            _
        ) = calculate_mean_and_std_of_multiple_exps(x_test_return_data_list, test_normalized_return_mean_data_list)
        all_results.append([
            mean_data, std_data, norm_mean_data, norm_std_data, common_timeline,
            test_mean_data, test_std_data, norm_test_mean_data, norm_test_std_data, test_common_timeline,
            n_samples,
            return_mean_data_list, normalized_return_mean_data_list,
            test_return_mean_data_list, test_normalized_return_mean_data_list
        ])

    # Create plots
    plot_title = env_name
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)
    create_multiple_exps_plot(
        all_results,
        path_to_save,
        plot_title,
        algo_names,
        env_name,
        plot_train=plot_train,
        plot_legend_bool=plot_legend_bool
    )

    # Create csv file with the mean best rewards
    calculate_mean_best_reward_over_multiple_experiments(
        all_results,
        path_to_save,
        algo_names,
        env_name
    )

    ## Save 'all_results' to use them for extracting average plots per algorithm aver all tasks of each benchmark
    pickle_file_path = os.path.join(path_to_save, f"all_results_env={env_name}.pkl")
    # Open a pickle file for writing
    with open(pickle_file_path, 'wb') as file:
        # Serialize the dictionary and write it to the file
        pickle.dump({"all_results_list": all_results, "algo_names": algo_names, "env_name": env_name}, file)

    print("\nMultiple-experiment plots created successfully! "
          f"\nSaved at: {path_to_save}")


def plot_average_per_algo_for_all_tasks_of_a_benchmark(
        paths_to_pickle_results,
        plot_title,
        path_to_save,
        plot_legend_bool
):
    """
    paths_to_pickle_results: are pickle files, each of which has 3 elements:
        - all_results_list
        - algo_names
        - env_name
    The 'all_results_list' element contains the results of each algo
    in the same order as in their names in 'algo_names'.
    For each algo, the following elements are stored in a list:
        0) mean_data
        1) std_data
        2) norm_mean_data
        3) norm_std_data
        4) common_timeline
        5) test_mean_data
        6) test_std_data
        7) norm_test_mean_data
        8) norm_test_std_data
        9) test_common_timeline
        10) n_samples
        11) return_mean_data_list
        12) normalized_return_mean_data_list
        13) test_return_mean_data_list
        14) test_normalized_return_mean_data_list
    """

    all_results_dict = {}
    all_common_timelines_dict = {}
    for path_to_pickle_task_results in paths_to_pickle_results:
        # Read the pickle file
        with open(path_to_pickle_task_results, 'rb') as file:
            pickle_data = pickle.load(file)
            all_algo_results = pickle_data['all_results_list']
            algo_names = pickle_data['algo_names']

        temp_max = -np.inf
        temp_min = np.inf
        for algo_name_idx, algo_name in enumerate(algo_names):
            algo_mean_results = all_algo_results[algo_name_idx][5]  # test mean data
            # Keep min and max to normalize the values across different tasks
            algo_max = np.max(algo_mean_results)
            algo_min = np.min(algo_mean_results)
            if temp_max < algo_max:
                temp_max = algo_max
            if temp_min > algo_min:
                temp_min = algo_min

        assert temp_min != np.inf, f'temp_min: {temp_min}'
        assert temp_max != -np.inf

        ## Re-iterate over algo results and normalize the values
        for algo_name_idx, algo_name in enumerate(algo_names):
            if algo_name not in list(all_results_dict.keys()):
                all_results_dict[algo_name] = []
            if algo_name not in list(all_common_timelines_dict.keys()):
                all_common_timelines_dict[algo_name] = []
            algo_mean_results = all_algo_results[algo_name_idx][5]  # test mean data
            # Normalize
            norm_algo_mean_results = (algo_mean_results - temp_min) / (temp_max - temp_min)
            # Store to dict
            all_results_dict[algo_name].append(norm_algo_mean_results.copy())
            # Store common_timeline to dict
            all_common_timelines_dict[algo_name].append(all_algo_results[0][9])  # test common timeline

    # Calculate the minimum common timeline for each algo
    for alg_name, alg_common_timelines in all_common_timelines_dict.items():
        temp_min_common_timeline = alg_common_timelines[0]
        for alg_common_timeline in alg_common_timelines:
            if len(temp_min_common_timeline) > len(alg_common_timeline):
                temp_min_common_timeline = alg_common_timeline
        all_common_timelines_dict[alg_name] = temp_min_common_timeline

    # Font size configuration
    title_fontsize = 18
    label_fontsize = 16

    # Set global parameters for font sizes
    mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick label font size
    mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick label font size

    # Create new figures, one for returns and another one for per-set returns.
    plt.figure()
    plt.xlabel("Steps", fontsize=label_fontsize)
    plt.ylabel("Normalized Episodic Reward", fontsize=label_fontsize)
    plt.title(plot_title, fontsize=title_fontsize)

    lines = []  # To keep track of plot lines for legend
    plot_legends = []  # To keep track of plot legends
    extra_lines = []  # To track lines not in PREDEFINED_MAP_ALGO_COLORS
    extra_plot_legends = []  # To track labels not in PREDEFINED_MAP_ALGO_COLORS

    for alg_name, alg_results in all_results_dict.items():

        # Define the label to plot in legend
        plot_legend = alg_name

        # Retrieve color based on 'plot_legend', or generate a random color if label is not in color_map
        color = PREDEFINED_MAP_ALGO_COLORS.get(plot_legend, f'#{random.randint(0, 0xFFFFFF):06x}')

        # Keep the used plot_legend
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            plot_legends.append(plot_legend)
        else:
            extra_plot_legends.append(plot_legend)

        # Keep the results according to the minimum common timeline
        min_common_time = min(min([len(x) for x in alg_results]), len(all_common_timelines_dict[alg_name]))
        alg_results = [alg_result[:min_common_time] for alg_result in alg_results]

        # Calculate mean
        mean_alg_results = np.mean(alg_results, axis=0)

        # Plot the test data
        line, = plt.plot(
            all_common_timelines_dict[alg_name][:min_common_time],
            mean_alg_results,
            label=plot_legend,
            color=color
        )

        # Append to either main or extra lines list
        if plot_legend in PREDEFINED_MAP_ALGO_COLORS:
            lines.append(line)
        else:
            extra_lines.append(line)

        ## Add std
        # Calculate standard deviation
        std_alg_results = np.std(alg_results, axis=0)
        # Calculate the upper and lower bounds of the standard deviation
        n_samples = len(alg_results)
        std_upper = mean_alg_results + 1.15*std_alg_results / np.sqrt(n_samples)  # 75%
        std_lower = mean_alg_results - 1.15*std_alg_results / np.sqrt(n_samples)  # 75%
        # Add a shaded area for the standard deviation
        plt.fill_between(all_common_timelines_dict[alg_name][:min_common_time], std_lower, std_upper, alpha=0.2, color=color)

    ## Custom legend creation
    legend_order = list(PREDEFINED_MAP_ALGO_COLORS.keys())
    legend_lines_fig = []
    # First, there should be the lines that are listed in 'legend_order'
    for _plot_legend in legend_order:
        if _plot_legend in plot_legends:
            # Find the index of '_plot_legend' in 'plot_legends'
            _plot_legend_idx = plot_legends.index(_plot_legend)
            # Add the lines for the figure
            legend_lines_fig.extend([lines[_plot_legend_idx]])
    # Then should be all the rest lines (of algorithms not listed in 'legend_order')
    for _line in extra_lines:
        # Add the lines for the figure
        legend_lines_fig.extend([_line])

    plt.tight_layout()
    if plot_legend_bool:
        _plot_custom_legend()

    # Save and close
    if os.path.exists(path_to_save) is False:
        os.makedirs(path_to_save)
    path_to_save_plot = os.path.join(path_to_save, f"benchmark={plot_title}")
    plt.savefig(path_to_save_plot, bbox_inches='tight')
    plt.close()

    # Resetting the following parameters to their default values
    mpl.rcParams['xtick.labelsize'] = mpl.rcParamsDefault['xtick.labelsize']
    mpl.rcParams['ytick.labelsize'] = mpl.rcParamsDefault['ytick.labelsize']

    print("\nAverage plots created successfully! "
          f"\nSaved at: {path_to_save_plot}")

def plot_mode(mode: str, config_path: str, config_data: dict):
    """
    Run the appropriate plotting function based on the selected mode.
    mode: str, one of 'single', 'multiple', 'average'.
    config_path: str, path to the config file.
    config_data: dict, the loaded config data from the YAML file.
    """
    try:
        if mode == 'single':
            print(f"Running in 'single' mode with config: {config_path}")
            plot_single_experiment_results(**config_data)

        elif mode == 'multiple':
            print(f"Running in 'multiple' mode with config: {config_path}")
            plot_multiple_experiment_results(**config_data)

        elif mode == 'average':
            print(f"Running in 'average' mode with config: {config_path}")
            plot_average_per_algo_for_all_tasks_of_a_benchmark(**config_data)
        else:
            print(f"Error: Unknown mode '{mode}'. Choose from 'single', 'multiple', or 'average'.",
                  file=sys.stderr)
            sys.exit(1)

    except TypeError as e:
        print(f"Error: Mismatch between config file keys and function arguments for mode '{mode}'.",
              file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """
    Main function to parse arguments and run the appropriate plotting function.
    Usage:
        python plot_utils.py <mode> --config <config_file.yaml>
    Modes:
        'single': Plot results for a single experiment run.
        'multiple': Plot aggregated results for multiple runs of multiple algorithms.
        'average': Plot average results across multiple tasks for a benchmark.
    """
    parser = argparse.ArgumentParser(description="Run plotting functions for MARL experiments.")
    parser.add_argument(
        "mode",
        choices=['single', 'multiple', 'average'],
        help=(
            "The plotting mode to run: \n"
            "'single': Plot results for a single experiment run.\n"
            "'multiple': Plot aggregated results for multiple runs of multiple algorithms.\n"
            "'average': Plot average results across multiple tasks for a benchmark.\n"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the YAML config file containing arguments for the chosen mode."
    )

    args = parser.parse_args()
    config_path = args.config
    if config_path is None:
        default_config_name = f"{args.mode}.yaml"
        config_path = os.path.join("config", "plots", default_config_name)
        print(f"No --config specified. Using default: {config_path}")

    config_path = os.path.expanduser(config_path)  # make the ~/file.yaml to /home/user/file.yaml as the full path is needed

    # Read config file
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Could not decode YAML from {config_path}", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    for key, value in config_data.items():
        if isinstance(value, str) and value.startswith('~/'):
            config_data[key] = os.path.expanduser(value)
        elif isinstance(value, list):
            config_data[key] = [
                os.path.expanduser(v) if isinstance(v, str) and v.startswith('~/') else v for v in value
            ]

    plot_mode(args.mode, config_path, config_data)

if __name__ == '__main__':
    main()
