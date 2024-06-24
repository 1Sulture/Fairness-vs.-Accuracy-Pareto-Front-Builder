import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from aif360.algorithms.postprocessing import RejectOptionClassification
from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# ************************************************ START CUSTOMIZATION ************************************************

# DATASET'S FILE NAME TO READ FROM (excluding .csv extension)
dataset_file_name = "adult.data_cleaned"

# WHETHER TO SHOW GRAPHS UPON COMPLETION
show_all_graphs = True

# WHETHER TO PRINT GRAPH TITLES
print_titles = True

# WHETHER TO PRINT GRAPH Y-AXIS LABELS
print_y_labels = True

# ************************************************ END CUSTOMIZATION ************************************************

# Metrics to evaluate
metrics = ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]


def load_data():
    """
    Loads the dataset from provided CSV file & prints its entry count.
    :return: the dataset as a pandas dataframe
    """
    # Load the data
    data = pd.read_csv(f'cleaned-datasets/{dataset_file_name}.csv')

    # Prep strings for printing & writing
    dataset_file_str = f"Imported file name: {dataset_file_name}.csv"
    entry_ct_pre_str = f"Data entry count: {len(data)}"

    # Print dataset stats & write to file
    print()
    print(dataset_file_str)
    print(entry_ct_pre_str)
    print()

    return data


def compute_metrics(f, metric_obj):
    """
    Computes all metrics on given computer; for printing to terminal and/or TXT file.
    :param f: the TXT file to write to
    :param metric_obj: object containing the dataset & model values to generate metric values from
    :return: all metrics in two combined OrderedDicts
    """
    # Accuracy metrics
    accuracy_metrics = OrderedDict()
    accuracy_metrics["True positive rate"] = metric_obj.true_positive_rate()
    accuracy_metrics["False positive rate"] = metric_obj.false_positive_rate()
    accuracy_metrics["True negative rate"] = metric_obj.true_negative_rate()
    accuracy_metrics["False negative rate"] = metric_obj.false_negative_rate()
    accuracy_metrics["Accuracy"] = metric_obj.accuracy()
    accuracy_metrics["Balanced accuracy"] = 0.5 * (metric_obj.true_positive_rate() +
                                                   metric_obj.true_negative_rate())
    accuracy_metrics["Precision"] = metric_obj.precision()
    accuracy_metrics["Recall"] = metric_obj.recall()

    # Fairness metrics
    fairness_metrics = OrderedDict()
    fairness_metrics["Demographic (statistical) parity difference"] = metric_obj.statistical_parity_difference()
    fairness_metrics["Disparate impact"] = metric_obj.disparate_impact()
    fairness_metrics["Average odds difference"] = metric_obj.average_odds_difference()
    fairness_metrics["Equalized odds difference"] = metric_obj.equalized_odds_difference()
    fairness_metrics["Equal opportunity difference"] = metric_obj.equal_opportunity_difference()
    fairness_metrics["Theil index"] = metric_obj.theil_index()

    # Write to provided TXT file
    f.write("\nAccuracy metrics:\n")
    for k in accuracy_metrics:
        f.write("%s = %.4f\n" % (k, accuracy_metrics[k]))

    f.write("\nFairness metrics:\n")
    for k in fairness_metrics:
        f.write("%s = %.4f\n" % (k, fairness_metrics[k]))

    return fairness_metrics


def process_dataset_roc(protected_attribute, data):
    """
    Runs the ROC post-processing on the loaded dataset with all 3 guiding metrics.
    Generates and saves a TXT file & graph PNG file to show the trends.
    :param protected_attribute: the protected attribute to consider
    """
    # Prepare data for AIF360 formatting
    train, test = train_test_split(data, test_size=0.3, stratify=data['income'])

    train = StandardDataset(train, label_name='income', favorable_classes=[1],
                            protected_attribute_names=[protected_attribute],
                            privileged_classes=[[1]]) # assuming 1 is the privileged class
    test = StandardDataset(test, label_name='income', favorable_classes=[1],
                           protected_attribute_names=[protected_attribute],
                           privileged_classes=[[1]])

    # Scale data
    scaler = MinMaxScaler(copy=False)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.transform(test.features)

    # Store fairness graph data
    fairness_graph_data_pre_roc = {metric: {"DIs": [], "DPs": [], "EOs": [], "AOs": []}
                                          for metric in metrics}
    fairness_graph_data_post_roc = {metric: {"DIs": [], "DPs": [], "EOs": [], "AOs": []}
                                           for metric in metrics}

    # Store accuracy graph data
    accuracy_graph_data_pre_roc = {metric: {"ACs": [], "B_ACs": [], "PCs": [], "RCs": []}
            for metric in metrics}
    accuracy_graph_data_post_roc = {metric: {"ACs": [], "B_ACs": [], "PCs": [], "RCs": []}
            for metric in metrics}

    # Set up file writing
    filename = f"{dataset_file_name}_roc_{protected_attribute}_log-reg"
    f = open(f"generated-txt-files/{filename}.txt", "w")
    f.write(f"data for {filename}.png\n\n")

    # Iterate over models
    for metric in metrics:
        print(f"Processing using metric: {metric}")

        index = train.feature_names.index(protected_attribute)
        X_tr = np.delete(train.features, index, axis=1)
        X_te = np.delete(test.features, index, axis=1)
        y_tr = train.labels.ravel()

        model = LogisticRegression(class_weight='balanced', solver='liblinear')
        model.fit(X_tr, y_tr)
        y_pred_proba = model.predict_proba(X_te)[:, 1].reshape(-1, 1)  # needed for ROC to work
        y_pred = model.predict(X_te)

        test_pred = test.copy()
        test_pred.scores = y_pred_proba  # needed for ROC to work
        test_pred.labels = y_pred

        # Giving protected attribute binary notation
        p = [{protected_attribute: 1}]
        u = [{protected_attribute: 0}]

        # THE FOLLOWING METRICS ARE COMPUTED FOR THE MODEL PRE-ROC

        # Add Disparate Impact before ROC
        cm = BinaryLabelDatasetMetric(test_pred, privileged_groups=p, unprivileged_groups=u)
        fairness_graph_data_pre_roc[metric]["DIs"].append(cm.disparate_impact())

        # Add remaining metrics before ROC
        cls_metric = ClassificationMetric(test, test_pred, privileged_groups=p, unprivileged_groups=u)
        fairness_graph_data_pre_roc[metric]["DPs"].append(cls_metric.statistical_parity_difference())
        fairness_graph_data_pre_roc[metric]["EOs"].append(cls_metric.equal_opportunity_difference())
        fairness_graph_data_pre_roc[metric]["AOs"].append(cls_metric.average_odds_difference())

        # Add accuracy params
        accuracy_graph_data_pre_roc[metric]["ACs"].append(cls_metric.accuracy())
        accuracy_graph_data_pre_roc[metric]["B_ACs"].append(
            0.5 * (cls_metric.true_positive_rate() + cls_metric.true_negative_rate()))
        accuracy_graph_data_pre_roc[metric]["PCs"].append(cls_metric.precision())
        accuracy_graph_data_pre_roc[metric]["RCs"].append(cls_metric.recall())

        # Print & save metrics before ROC (only need to do once)
        if metric == "Statistical parity difference":
            f.write("Metrics BEFORE ROC:")
            compute_metrics(f, cls_metric)

        # Apply Reject Option Classification (ROC)
        roc = RejectOptionClassification(unprivileged_groups=u, privileged_groups=p, low_class_thresh=0.01,
                                         high_class_thresh=0.99, num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=metric, metric_ub=0.05, metric_lb=-0.05)
        roc = roc.fit(test, test_pred)
        test_repd_pred = roc.predict(test_pred)

        # THE FOLLOWING METRICS ARE COMPUTED FOR THE MODEL POST-ROC

        # Add Disparate Impact after ROC
        cm = BinaryLabelDatasetMetric(test_repd_pred, privileged_groups=p, unprivileged_groups=u)
        fairness_graph_data_post_roc[metric]["DIs"].append(cm.disparate_impact())

        # Add remaining metrics after ROC
        cls_metric = ClassificationMetric(test, test_repd_pred, privileged_groups=p, unprivileged_groups=u)
        fairness_graph_data_post_roc[metric]["DPs"].append(cls_metric.statistical_parity_difference())
        fairness_graph_data_post_roc[metric]["EOs"].append(cls_metric.equal_opportunity_difference())
        fairness_graph_data_post_roc[metric]["AOs"].append(cls_metric.average_odds_difference())

        # Add accuracy params
        accuracy_graph_data_post_roc[metric]["ACs"].append(cls_metric.accuracy())
        accuracy_graph_data_post_roc[metric]["B_ACs"].append(
            0.5 * (cls_metric.true_positive_rate() + cls_metric.true_negative_rate()))
        accuracy_graph_data_post_roc[metric]["PCs"].append(cls_metric.precision())
        accuracy_graph_data_post_roc[metric]["RCs"].append(cls_metric.recall())

        # Print & save metrics after ROC (for each guiding metric)
        f.write(f"\n\nMetrics AFTER ROC - {metric}:")
        compute_metrics(f, cls_metric)

    # Plot results
    generate_graphs_and_txt(protected_attribute, filename, fairness_graph_data_pre_roc,
                            accuracy_graph_data_pre_roc, fairness_graph_data_post_roc, accuracy_graph_data_post_roc)


def generate_graphs_and_txt(protected_attribute, filename, fairness_graph_data_pre_roc, accuracy_graph_data_pre_roc,
                            fairness_graph_data_post_roc, accuracy_graph_data_post_roc):
    """
    Generates and saves graphs showing how 4 fairness metrics change when training ROC with 3 different guiding metrics.
    :param protected_attribute: the dataset's protected attribute
    :param filename: the file name for the PNG file
    :param fairness_graph_data_pre_roc: fairness graph data - before ROC
    :param accuracy_graph_data_pre_roc: accuracy graph data - before ROC
    :param fairness_graph_data_post_roc: fairness graph data - after ROC
    :param accuracy_graph_data_post_roc: accuracy graph data - after ROC
    """

    print()
    print("Generating graphs")

    plt.layout = 'constrained'
    plt.style.use('grayscale')
    plt.figure(figsize=(6, 9))
    line_styles = ["--", "-.", ":"]

    # Iterate through all metrics that guided the ROC
    for j, metric in enumerate(metrics):
        x_values = ["Pre-ROC", "Post-ROC"]

        # Disparate impact graph
        plt.subplot(421)
        if print_titles:
            plt.title("Disparate Impact")
        if print_y_labels:
            plt.ylabel('Disparate impact level', fontsize='medium')
        y_values = [
            fairness_graph_data_pre_roc[metric]["DIs"],
            fairness_graph_data_post_roc[metric]["DIs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [1, 1], 'b')
        plt.plot([0, 1], [0.8, 0.8], 'g')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Statistical parity difference graph
        plt.subplot(422)
        if print_titles:
            plt.title("Statistical Parity Difference")
        if print_y_labels:
            plt.ylabel('Statistical Parity Difference', fontsize='medium')
        y_values = [
            fairness_graph_data_pre_roc[metric]["DPs"],
            fairness_graph_data_post_roc[metric]["DPs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [0, 0], 'b')
        plt.plot([0, 1], [-0.05, -0.05], 'g')
        plt.plot([0, 1], [-0.1, -0.1], 'yellowgreen')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Equal opportunity graph
        plt.subplot(423)
        if print_titles:
            plt.title("Equal Opportunity")
        if print_y_labels:
            plt.ylabel('Equal opportunity', fontsize='medium')
        y_values = [
            fairness_graph_data_pre_roc[metric]["EOs"],
            fairness_graph_data_post_roc[metric]["EOs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [-0.1, -0.1], 'yellowgreen')
        plt.plot([0, 1], [-0.05, -0.05], 'g')
        plt.plot([0, 1], [0, 0], 'b')
        plt.plot([0, 1], [0.05, 0.05], 'g')
        # plt.plot([0, 1], [0.1, 0.1], 'yellowgreen')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Equal opportunity graph
        plt.subplot(424)
        if print_titles:
            plt.title("Average Odds Difference")
        if print_y_labels:
            plt.ylabel('Average odds difference', fontsize='medium')
        y_values = [
            fairness_graph_data_pre_roc[metric]["AOs"],
            fairness_graph_data_post_roc[metric]["AOs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [-0.1, -0.1], 'yellowgreen')
        plt.plot([0, 1], [-0.05, -0.05], 'g')
        plt.plot([0, 1], [0, 0], 'b')
        plt.plot([0, 1], [0.05, 0.05], 'g')
        # plt.plot([0, 1], [0.1, 0.1], 'yellowgreen')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Accuracy graph
        plt.subplot(425)
        if print_titles:
            plt.title("Accuracy")
        if print_y_labels:
            plt.ylabel('Accuracy', fontsize='medium')
        y_values = [
            accuracy_graph_data_pre_roc[metric]["ACs"],
            accuracy_graph_data_post_roc[metric]["ACs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [1, 1], 'b')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Balanced Accuracy graph
        plt.subplot(426)
        if print_titles:
            plt.title("Balanced Accuracy")
        if print_y_labels:
            plt.ylabel('Balanced accuracy', fontsize='medium')
        y_values = [
            accuracy_graph_data_pre_roc[metric]["B_ACs"],
            accuracy_graph_data_post_roc[metric]["B_ACs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [1, 1], 'b')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

        # Precision graph
        plt.subplot(427)
        if print_titles:
            plt.title("Precision")
        if print_y_labels:
            plt.ylabel('Precision', fontsize='medium')
        y_values = [
            accuracy_graph_data_pre_roc[metric]["PCs"],
            accuracy_graph_data_post_roc[metric]["PCs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [1, 1], 'b')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")
        plt.legend(loc='center', framealpha=1.0)

        # Recall graph
        plt.subplot(428)
        if print_titles:
            plt.title("Recall")
        if print_y_labels:
            plt.ylabel('Recall', fontsize='medium')
        y_values = [
            accuracy_graph_data_pre_roc[metric]["RCs"],
            accuracy_graph_data_post_roc[metric]["RCs"]
        ]
        plt.plot(x_values, y_values, marker='o', label=metric, linestyle=line_styles[j])
        plt.plot([0, 1], [1, 1], 'b')
        for x, y in zip(x_values, y_values):
            plt.text(x, y[0], f"{y[0]:.3f}", ha='center', va='bottom', fontsize="small")

    plt.suptitle(f"Protected attribute: {protected_attribute}")
    plt.tight_layout()
    plt.savefig(f"generated-pics/{filename}.png")
    if show_all_graphs:
        plt.show()

    print("Graphs generated successfully")
    print()


# **** BASE CODE STARTS HERE ****

# Load prepared dataset
data = load_data()

# Map of protected attributes to test
protected_attrs = ['sex', 'race']

# Process each protected attribute config of dataset
for i, prot_attr in enumerate(protected_attrs):
    print(f"\nPROCESSING WITH '{prot_attr}' AS SENSITIVE ATTRIBUTE")
    process_dataset_roc(prot_attr, data)

# Indicate success!
print()
print("Operation successful")


