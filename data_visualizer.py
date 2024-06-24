import numpy as np
from matplotlib import pyplot as plt
import csv

from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ************************************************ START CUSTOMIZATION ************************************************

# **** FOR ALL GRAPHS ****

# FILE NAME OF CSV FILE TO READ FROM (excluding .csv extension)
dataset_file_name = ""

# PROTECTED ATTRIBUTE TO EXAMINE (only for labeling, not functionality)
#  ("race" or "sex")
selected_protected_attr = ""

# WHETHER TO SHOW GRAPHS UPON GENERATION
show_new_graphs = True

# **** FOR METRIC GRAPHS ONLY ****

# WHETHER TO GENERATE METRIC GRAPHS
generate_metric_graphs = True

# WHETHER TO GENERATE TITLE
generate_metric_graph_titles = True
generate_metric_graph_super_title = True

# WHETHER TO DISPLAY Y-AXIS LABEL
generate_metric_graph_y_label = True

# FAIRNESS METRICS TO SHOW:
selected_fair_metric_names = ["Statistical parity difference", "Disparate impact", "Average odds difference",
                         "Equalized odds difference", "Equal opportunity difference", "Theil index"]
# ACCURACY METRICS TO SHOW:
selected_acc_metric_names = ["True positive rate", "False positive rate", "True negative rate", "False negative rate",
                         "Accuracy", "Balanced accuracy", "Precision", "Recall"]

# **** FOR PARETO GRAPHS ONLY ****

# WHETHER TO GENERATE PARETO FRONT
generate_pareto_fronts = True

# WHETHER TO MENTION PROTECTED ATTRIBUTE IN TITLE
protected_attr_in_title = True

# FAIRNESS METRIC (0 to 5):
fairness_metric_index = 0

# ACCURACY METRIC (0 to 7):
accuracy_metric_index = 4

# FAIRNESS WEIGHT (default = 1.0)
lambda_weight = 1.0

# ************************************************ END CUSTOMIZATION ************************************************

# List of metrics we track
# NOTE: "True positive rate" and "Recall" are the same!
fairness_metric_names = ["Statistical parity difference", "Disparate impact", "Average odds difference",
                         "Equalized odds difference", "Equal opportunity difference", "Theil index"]
accuracy_metric_names = ["True positive rate", "False positive rate", "True negative rate", "False negative rate",
                         "Accuracy", "Balanced accuracy", "Precision", "Recall"]

# Abbreviated versions (for file naming)
fairness_metric_names_abbrev = ["stat-par-diff", "disp-imp", "avg-odd-diff", "eq-odd-diff", "eq-oppo-diff", "th-index"]
accuracy_metric_names_abbrev = ["tr-pos-rate", "fa-pos-rate", "tr-neg-rate", "fa-neg-rate", "acc", "bal-acc", "prec",
                                "recall"]

# Names of ROC stages for indexing and clarity
roc_labels = ["Pre-ROC", "Post-ROC"]

# Set up metric data
pareto_fairness_metric = fairness_metric_names[fairness_metric_index]
pareto_accuracy_metric = accuracy_metric_names[accuracy_metric_index]
pareto_fairness_metric_abbrev = fairness_metric_names_abbrev[fairness_metric_index]
pareto_accuracy_metric_abbrev = accuracy_metric_names_abbrev[accuracy_metric_index]


def read_metrics_from_csv(file_name):
    """
    Reading CSV file of specified file name containing all fairness & accuracy metric values we track
    :param file_name: name of file to read
    :return: master dict containing all metric data
    """
    metric_dict_from_csv = [{"Accuracy metrics": [], "Fairness metrics": []},
                            {"Accuracy metrics": [], "Fairness metrics": []}]

    # Initialize lists for each level of repair in each ROC stage
    for roc_stage in metric_dict_from_csv:
        # Set up dict to hold values for metrics at all 11 levels
        for i in range(11):
            roc_stage["Accuracy metrics"].append({metric: 0.0 for metric in accuracy_metric_names})
            roc_stage["Fairness metrics"].append({metric: 0.0 for metric in fairness_metric_names})

    with open(file_name, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip the header row

        # Read for each repair level & ROC stage (11 repair levels * 2 stages = 22 rows)
        for row in reader:
            repair_level_index = int(float(row[0]) * 10)  # Convert repair level back to index (0-10)
            roc_stage_index = int(row[1])  # ROC stage (0 or 1)

            # Slice to get metrics from the row, considering the first two columns are repair level and ROC?
            accuracy_values = row[2:2 + len(accuracy_metric_names)]
            fairness_values = row[2 + len(accuracy_metric_names):]

            # Populate the metrics for the current repair level and ROC stage
            accuracy_metrics = metric_dict_from_csv[roc_stage_index]["Accuracy metrics"][repair_level_index]
            fairness_metrics = metric_dict_from_csv[roc_stage_index]["Fairness metrics"][repair_level_index]

            # Import data from each row and write to dict
            for i, metric_name in enumerate(accuracy_metric_names):
                accuracy_metrics[metric_name] = float(accuracy_values[i])
            for i, metric_name in enumerate(fairness_metric_names):
                fairness_metrics[metric_name] = float(fairness_values[i])

    return metric_dict_from_csv


def plot_selected_metric_data(metric_dict_master, file_name, metric_type_str, selected_metric_names):
    """
    Plots given metric data and saves to a PNG file with specified file name.
    :param metric_dict_master: the master dict with all metric data
    :param file_name: name for graph file
    :param metric_type_str: metric type: "Fairness metrics" or "Accuracy metrics"
    :param selected_metric_names: list of metric names to plot
    """
    # Initial graph set up
    x_values = np.linspace(0, 1, 11)  # DI repair levels from 0 to 1 in steps of 0.1
    x_label = "DIR Repair Level"

    best_fit_line_color = '#ff9999'

    # Set variables depending on metric type
    if metric_type_str == "Fairness metrics":
        metric_file_name_str = "fairness"
        point_rounding_lvl = 2
    else:
        metric_file_name_str = "accuracy"
        point_rounding_lvl = 3

    # Prepare subplot grid
    metric_ct = len(selected_metric_names)
    plt.rc('xtick', labelsize=13)  # adjust x tick font size
    plt.rc('ytick', labelsize=13)  # adjust y tick font size
    fig, axes = plt.subplots(2, metric_ct, figsize=(4 * metric_ct, 6.5))

    # For each ROC stage (pre & post)
    for roc_stage_index, roc_stage_data in enumerate(metric_dict_master):
        roc_label = roc_labels[roc_stage_index]

        # For each selected metric
        for i, metric_name in enumerate(selected_metric_names):
            # Safeguard in case only 1 metric selected
            if metric_ct == 1:
                ax = axes[roc_stage_index]
            else:
                ax = axes[roc_stage_index, i]

            # Plot y-values; annotate each with respective value
            y_values = [roc_stage_data[metric_type_str][lvl][metric_name] for lvl in range(11)]
            ax.plot(x_values, y_values, marker='o', color='black')
            for x, y in zip(x_values, y_values):
                ax.text(x, y, f"{y:.{point_rounding_lvl}f}", ha='center', va='bottom', fontsize="small",
                        bbox=dict(boxstyle="round,pad=0", facecolor='white', edgecolor='none', alpha=0.6))

            # If plotting fairness metrics, add additional colored lines to show performance
            if metric_type_str == "Fairness metrics":
                if metric_name == "Disparate impact":
                    ax.plot([0, 1], [1.2, 1.2], 'g')
                    ax.plot([0, 1], [1, 1], 'b')
                    ax.plot([0, 1], [0.8, 0.8], 'g')
                    # if roc_stage_index == 1:
                    #     ax.plot([0, 1], [1, 1], 'b')
                    # ax.plot([0, 1], [0.8, 0.8], 'g')
                else:
                    ax.plot([0, 1], [0.1, 0.1], 'yellowgreen')
                    ax.plot([0, 1], [0.05, 0.05], 'g')
                    ax.plot([0, 1], [0, 0], 'b')
                    ax.plot([0, 1], [-0.05, -0.05], 'g')
                    ax.plot([0, 1], [-0.1, -0.1], 'yellowgreen')
                    # if roc_stage_index == 1:
                    #     ax.plot([0, 1], [0, 0], 'b')
                    #     ax.plot([0, 1], [-0.05, -0.05], 'g')
                    # else:
                    #     ax.plot([0, 1], [-0.1, -0.1], 'yellowgreen')

            # Calculate and plot line of best fit
            coeffs = np.polyfit(x_values, y_values, 1)  # Linear fit
            poly_eq = np.poly1d(coeffs)
            y_fit = poly_eq(x_values)
            ax.plot(x_values, y_fit, best_fit_line_color)  # Plot the line of best fit with a red dashed line

            # Adjust tick sizes & formatting for readability
            if metric_type_str == "Accuracy metrics":
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # to avoid axis clutter

            # Set labeling
            ax.set_xticks(np.arange(0.0, 1.01, 0.2))
            ax.set_xlabel(x_label, fontsize=13)
            if generate_metric_graph_y_label:
                ax.set_ylabel(metric_name, fontsize=13)
            if generate_metric_graph_titles:
                ax.set_title(f'{roc_label} - {metric_name}', fontsize=14)

    # Customization for graph title
    if generate_metric_graph_super_title:
        plt.suptitle(
            f"Analysis for {metric_file_name_str} metrics across DIR repair levels for '{selected_protected_attr}' "
            f"attribute")
    plt.tight_layout()

    plt.savefig(f"generated-pics/{file_name}_{metric_file_name_str}-data-graphs.png")
    if show_new_graphs:
        plt.show()


def plot_metric_dict_data(metric_dict_master, file_name, selected_fairness_metric_names, selected_accuracy_metric_names,
                          plot_fairness=True, plot_accuracy=True):
    """
    Plots metric data in selection arrays and saves to a PNG file with specified file name.

    :param metric_dict_master: master dict containing all metric data
    :param file_name: name for graph file
    :param selected_fairness_metric_names: list of fairness metric names to plot
    :param selected_accuracy_metric_names: list of accuracy metric names to plot
    :param plot_fairness: boolean flag to plot fairness metrics
    :param plot_accuracy: boolean flag to plot accuracy metrics
    """

    # Fairness metric plotting
    if plot_fairness:
        print("Plotting fairness metric data")
        plot_selected_metric_data(metric_dict_master, file_name, "Fairness metrics",
                                  selected_fairness_metric_names)

    # Accuracy metric plotting
    if plot_accuracy:
        print("Plotting accuracy metric data")
        plot_selected_metric_data(metric_dict_master, file_name, "Accuracy metrics",
                                  selected_accuracy_metric_names)

    print("Data plotted & graphs saved successfully\n")


def zip_metrics(metric_dict_master, fairness_metric, accuracy_metric):
    """
    Zips values from specified fairness metric with specified accuracy metric, using data from master dict.
    :param metric_dict_master: the master dict containing all metric data
    :param fairness_metric: fairness metric to import data from
    :param accuracy_metric: accuracy metric to import data from
    :return: dict with 2 lists of fairness-accuracy data pairs, for pre- & post-ROC respectively
    """
    # Create a nested dictionary to hold the zipped metrics
    zipped_metrics = {"Pre-ROC": [], "Post-ROC": []}

    # Loop through the metric dictionary for both pre- & post-ROC data
    for roc_index, roc_stage in enumerate(roc_labels):
        # Access the specific ROC stage data
        roc_data = metric_dict_master[roc_index]

        # Iterate over all repair levels (0 to 10)
        for repair_level in range(11):
            # Fetch the required fairness and accuracy metrics for the current repair level
            fairness_value = roc_data["Fairness metrics"][repair_level][fairness_metric]
            accuracy_value = roc_data["Accuracy metrics"][repair_level][accuracy_metric]

            # Append a tuple of the selected metrics to the corresponding ROC stage in the zipped_metrics dict
            zipped_metrics[roc_stage].append((fairness_value, accuracy_value))

    return zipped_metrics


def is_dominated(x, others):
    """
    Checks if the provided point (pair) is Pareto-dominated by another point.
    :param x: point to be checked
    :param others: all other points to compare against
    :return: True if point is Pareto-dominated, False otherwise
    """
    for other in others:
        if (other[0] >= x[0] and other[1] > x[1]) or (other[0] > x[0] and other[1] >= x[1]):
            return True
    return False


def pareto_front(points):
    """
    Computes the Pareto front for a given list of points.
    :param points: list of tuples, where each tuple represents (fairness, accuracy) point pair
    :return: all points that would be the Pareto-front of the given point list
    """
    pareto_points = []
    for point in points:
        if not is_dominated(point, points):
            pareto_points.append(point)
    return pareto_points


def get_overall_best_point(best_points):
    """
    Gets best point between pre- & post-ROC stages and returns its respective stage.
    :param best_points: the 2 best points, given as pairs
    :return: the optimal ROC stage
    """
    # Initialize variables to track the best overall point
    best_score = float('-inf')
    # best_overall_point = None
    best_stage = None

    # Iterate through each ROC stage to find the best point
    for stage, data in best_points.items():
        if data['Score'] > best_score:
            best_score = data['Score']
            # best_overall_point = data['Point']
            best_stage = stage

    return best_stage


def plot_pareto_front_pair(data, pareto_points, file_name):
    """
    Plots all points for trade-off between 2 metrics, with 2 Pareto fronts for pre- & post-ROC respectively.
    :param data: all data points for both pre- & post-ROC stages
    :param pareto_points: only the Pareto-front points of pre- & post-ROC stages
    :param file_name: name of output file
    """
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Containers to store legend labels & best points of each ROC stage
    handles, labels = None, None
    best_points = {roc_stage: {"Point": [], "Score": 0.0} for roc_stage in roc_labels}

    # Plot for each ROC stage
    for i in range(2):
        ax = axes[i]
        roc_label = roc_labels[i]

        # Convert lists of tuples to numpy arrays for easier handling
        imported_data = np.array(data[i])
        imported_pareto_points = np.array(pareto_points[i])

        # Calculate accuracy score: inverted only for false positive/negative rates
        if pareto_accuracy_metric == "False positive rate" or pareto_accuracy_metric == "False negative rate":
            accuracy_score = 1 - imported_pareto_points[:, 1]
        else:
            accuracy_score = imported_pareto_points[:, 1]

        # Calculate fairness score with weight: linear absolute distance from fairness metric's optimal value
        fairness_score = lambda_weight * np.abs(
            imported_pareto_points[:, 0] - (1 if pareto_fairness_metric == "Disparate impact" else 0))

        # Calculate trade-off scores establish ranking
        scores = accuracy_score - fairness_score
        rankings = scores.argsort()[::-1].argsort() + 1  # calculate rankings from scores

        # Sort Pareto points for plotting based on the fairness value
        indices_sorted = np.argsort(imported_pareto_points[:, 0])
        imported_pareto_points_sorted = imported_pareto_points[indices_sorted]
        rankings_sorted = rankings[indices_sorted]

        # Plot all points and the Pareto front
        ax.scatter(imported_data[:, 0], imported_data[:, 1], color='blue', label='All Points')
        ax.plot(imported_pareto_points_sorted[:, 0], imported_pareto_points_sorted[:, 1], color='red',
                        marker='o', label='Pareto Front')

        # Annotate each Pareto point with its DI repair level and ranking
        for idx, point, rank in zip(indices_sorted, imported_pareto_points_sorted, rankings_sorted):
            ax.annotate(f'Level: {idx * 0.1:.1f}\nRank: {rank}',
                        xy=(point[0], point[1]), xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='darkgreen',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='none', alpha=0.5))

        # Highlight the best-ranked point & save it
        best_rank_index = np.argmin(rankings)
        best_point = imported_pareto_points[best_rank_index]
        best_points[roc_label]["Point"] = list(best_point)
        best_points[roc_label]["Score"] = scores[best_rank_index]

        # Print which point is best for given ROC stage
        print(f"{roc_label} best point:")
        print(f"Fairness: {best_point[0]:.4f}")
        print(f"Accuracy: {best_point[1]:.4f}")
        print()

        # Annotate the best point
        ax.annotate('Best\nFair: {:.3f}\nAcc: {:.3f}'.format(best_point[0], best_point[1]),
                    xy=(best_point[0], best_point[1]), xytext=(-60, -40), textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='green'),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', edgecolor='none', alpha=0.7))

        # Adjust titles and labels
        # ax.set_title(f'{roc_label}:\n{pareto_fairness_metric} vs. {pareto_accuracy_metric}\n')
        ax.set_title(f'{roc_label}\n', fontsize=14)
        ax.set_xlabel(pareto_fairness_metric, wrap=True, fontsize=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))  # to avoid axis clutter
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))  # to avoid axis clutter
        ax.set_ylabel(pareto_accuracy_metric, fontsize=12)
        ax.grid(True)

        # Save legend data
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()

    # Set up legend
    fig.legend(handles, labels, loc='lower center')

    # Adjust protected attribute label if necessary
    if selected_protected_attr == "sex":
        final_prot_attr_name = "gender"
    else:
        final_prot_attr_name = "race"

    # Print best point of both configs
    best_point_stage = get_overall_best_point(best_points)
    print(f"Best point is in {best_point_stage} stage")

    # Plot & save graph
    if protected_attr_in_title:
        fig.suptitle(f"Pareto Fronts for '{final_prot_attr_name}' attribute - "
                     f"{pareto_fairness_metric} vs. {pareto_accuracy_metric}\nFairness weight = {lambda_weight}"
                     f"\nBest point in {best_point_stage} stage")
    else:
        fig.suptitle(f"{pareto_fairness_metric} vs. {pareto_accuracy_metric} - Fairness weight = {lambda_weight}"
                     f"\nBest point in {best_point_stage} stage")
    fig.tight_layout()
    fig.savefig(f'generated-pics/{file_name}_pareto-front_'
                f'{pareto_fairness_metric_abbrev}_{pareto_accuracy_metric_abbrev}_fair-wt={lambda_weight}.png', dpi=300)
    if show_new_graphs:
        plt.show()


def generate_pareto_pair(metric_dict_master, file_name):
    """
    Computes Pareto points given the selected fairness & accuracy metrics and then generates and saves their graphs.
    :param metric_dict_master: master dict containing all metric data
    :param file_name: file name for graph PNG file
    """
    print("Generating Pareto fronts")
    print("Fairness metric: " + pareto_fairness_metric)
    print("Accuracy metric: " + pareto_accuracy_metric)
    print()

    # Generate data pairs between selected fairness & accuracy metrics
    zipped_fair_acc_vals = zip_metrics(metric_dict_master, pareto_fairness_metric, pareto_accuracy_metric)

    # Gather data points and find pareto points for both ROC configs
    data_points = [zipped_fair_acc_vals["Pre-ROC"], zipped_fair_acc_vals["Post-ROC"]]
    pareto_points = [pareto_front(data_points[0]), pareto_front(data_points[1])]

    # Plot pareto fronts
    plot_pareto_front_pair(data_points, pareto_points, file_name)
    print("Data plotted & graphs saved successfully")


# **** BASE CODE STARTS HERE ****

# Load data
loaded_metric_dict = read_metrics_from_csv(f"generated-csv-files/{dataset_file_name}.csv")

print(f"Generating for '{selected_protected_attr}' attribute\n")

# Plot data if requested
if generate_metric_graphs:
    plot_metric_dict_data(loaded_metric_dict, dataset_file_name, selected_fair_metric_names, selected_acc_metric_names)

# Plot Pareto fronts if requested:
if generate_pareto_fronts:
    generate_pareto_pair(loaded_metric_dict, dataset_file_name)
