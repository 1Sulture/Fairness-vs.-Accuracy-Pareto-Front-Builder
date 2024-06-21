from collections import OrderedDict

import numpy as np
import pandas as pd
from aif360.algorithms.postprocessing import RejectOptionClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.datasets import StandardDataset
from aif360.metrics import ClassificationMetric

# ************************************************ START CUSTOMIZATION ************************************************

# DATASET'S FILE NAME TO READ FROM (excluding .csv extension)
dataset_file_name = "adult.data_cleaned"

# ROC METRIC TO USE (0 to 2)
# 0 = "Statistical parity difference" (SPD)
# 1 = "Average odds difference" (AOD)
# 2 = "Equal opportunity difference" (EOD)
roc_metric_index = 0

# MODEL TO USE (0 to 5)
# 0 = "Logistic Regression"
# 1 = "K-Nearest Neighbors"
# 2 = "Support Vector Machine"
# 3 = "Decision Tree"
# 4 = "Random Forest"
# 5 = "Neural Network"
selected_model_index = 0

# FOR KNN, NUMBER OF NEIGHBORS
knn_neighbor_ct = 7

# ************************************************ END CUSTOMIZATION ************************************************

# Set selected model & abbreviation (for file naming)
model_names = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Support Vector Machine",
    "Decision Tree",
    "Random Forest",
    "Neural Network",
]
model_abbrev = ["log-reg", f"knn-{knn_neighbor_ct}", "svm", "dec-tr", "rand-fst", "nn"]
selected_model_name = model_names[selected_model_index]
selected_model_abbrev = model_abbrev[selected_model_index]

# List of metrics we track
# NOTE: "True positive rate" and "Recall" are the same!
accuracy_metric_names = ["True positive rate", "False positive rate", "True negative rate", "False negative rate",
                         "Accuracy", "Balanced accuracy", "Precision", "Recall"]
fairness_metric_names = ["Statistical parity difference", "Disparate impact", "Average odds difference",
                         "Equalized odds difference", "Equal opportunity difference", "Theil index"]

# Initialize the dictionary to store metric data
metric_dict_master = [{"Accuracy metrics": [], "Fairness metrics": []},
                      {"Accuracy metrics": [], "Fairness metrics": []}]

# Set selected ROC metric
roc_metrics = ["Statistical parity difference", "Average odds difference", "Equal opportunity difference"]
roc_metrics_abbrev = ["spd", "aod", "rod"]
roc_metric = roc_metrics[roc_metric_index]
roc_metric_abbrev = roc_metrics_abbrev[roc_metric_index]

# Set up the structure for each repair level from 0.0 to 1.0 in steps of 0.1
for roc_stage in metric_dict_master:
    for i in range(11):  # for 11 levels (0 to 1.0)
        roc_stage["Accuracy metrics"].append({metric: 0.0 for metric in accuracy_metric_names})
        roc_stage["Fairness metrics"].append({metric: 0.0 for metric in fairness_metric_names})


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


def save_metrics_to_dict(roc_stage_val, repair_lvl, metric_obj):
    """
    Computing & saving metric values from given metric object.
    :param roc_stage_val: string indicating whether we're pre- or post-ROC
    :param repair_lvl: DIR repair level index (0 to 10)
    :param metric_obj: object containing the dataset & model values to generate metric values from
    """
    # Accuracy metrics
    accuracy_metrics = metric_dict_master[roc_stage_val]["Accuracy metrics"][repair_lvl]
    accuracy_metrics["True positive rate"] = metric_obj.true_positive_rate()
    accuracy_metrics["False positive rate"] = metric_obj.false_positive_rate()
    accuracy_metrics["True negative rate"] = metric_obj.true_negative_rate()
    accuracy_metrics["False negative rate"] = metric_obj.false_negative_rate()
    accuracy_metrics["Accuracy"] = metric_obj.accuracy()
    accuracy_metrics["Balanced accuracy"] = 0.5 * (metric_obj.true_positive_rate() + metric_obj.true_negative_rate())
    accuracy_metrics["Precision"] = metric_obj.precision()
    accuracy_metrics["Recall"] = metric_obj.recall()

    # Fairness metrics
    fairness_metrics = metric_dict_master[roc_stage_val]["Fairness metrics"][repair_lvl]
    fairness_metrics["Statistical parity difference"] = metric_obj.statistical_parity_difference()
    fairness_metrics["Disparate impact"] = metric_obj.disparate_impact()
    fairness_metrics["Average odds difference"] = metric_obj.average_odds_difference()
    fairness_metrics["Equalized odds difference"] = metric_obj.equalized_odds_difference()
    fairness_metrics["Equal opportunity difference"] = metric_obj.equal_opportunity_difference()
    fairness_metrics["Theil index"] = metric_obj.theil_index()


def write_metrics_to_csv(file_name):
    """
    Writes all generated metric values to CSV file of specified file name.
    :param file_name: name of file to write to
    """
    print(f"Writing metrics to csv file with name: {file_name}")

    import csv
    with open(f'generated-csv-files/{file_name}.csv', 'w', newline='') as csvfile:
        # Set up CSV writer for data
        writer = csv.writer(csvfile)
        column_names = ["Repair level (0-10)", "ROC?"] + accuracy_metric_names + fairness_metric_names
        writer.writerow(column_names)

        # Master loop for going through pre- & post-ROC stat dicts
        for roc_table_val, metric_dict in enumerate(metric_dict_master):
            # Loop for writing data for each row
            for rep_lvl in range(11):
                accuracy_metrics = metric_dict["Accuracy metrics"][rep_lvl]
                fairness_metrics = metric_dict["Fairness metrics"][rep_lvl]

                # Prepare data to be written for each repair level
                row_values = [rep_lvl / 10.0, roc_table_val]  # repair level & ROC status
                row_values += [accuracy_metrics[metric] for metric in accuracy_metric_names]  # accuracy metrics
                row_values += [fairness_metrics[metric] for metric in fairness_metric_names]  # fairness metrics

                writer.writerow(row_values)

    print("Metrics written successfully to csv file")


def write_metrics_to_txt(f, metrics):
    """
    Writes all generated metric values to specified TXT file.
    :param f: the file to write to
    :param metrics: the metrics to write
    """
    # Write all accuracy metrics
    f.write("\nAccuracy metrics:\n")
    for k, (name, value) in enumerate(metrics["Accuracy metrics"].items()):
        f.write("%s = %.4f\n" % (name, value))

    # Write all fairness metrics
    f.write("\nFairness metrics:\n")
    for k, (name, value) in enumerate(metrics["Fairness metrics"].items()):
        f.write("%s = %.4f\n" % (name, value))


def compute_metrics_for_txt(f, metric_obj):
    """
    Computes all metrics on given computer; for printing to TXT file.
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
    fairness_metrics["Statistical parity difference"] = metric_obj.statistical_parity_difference()
    fairness_metrics["Disparate impact"] = metric_obj.disparate_impact()
    fairness_metrics["Average odds difference"] = metric_obj.average_odds_difference()
    fairness_metrics["Equalized odds difference"] = metric_obj.equalized_odds_difference()
    fairness_metrics["Equal opportunity difference"] = metric_obj.equal_opportunity_difference()
    fairness_metrics["Theil index"] = metric_obj.theil_index()

    # Write all metric values to TXT file
    metric_dict = {"Fairness metrics": fairness_metrics, "Accuracy metrics": accuracy_metrics}
    write_metrics_to_txt(f, metric_dict)


def process_dataset_dir_roc(protected_attr, data):
    """
    Master function for running DIR pre-processing at different repair levels and ROC post-processing.
    This version also prints the generated data.
    :param protected_attr: the protected attribute to consider
    """
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', solver='liblinear'),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "Support Vector Machine": SVC(class_weight='balanced', kernel='linear', probability=True),
        "Neural Network": None,  # Placeholder, will be created dynamically
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=knn_neighbor_ct, weights='uniform')
    }

    # Set up file writing
    filename = f"{dataset_file_name}_dir-roc-{roc_metric_abbrev}_{protected_attr}_{selected_model_abbrev}"
    f = open(f"generated-txt-files/{filename}.txt", "w")
    f.write(f"data from {filename}.csv\n\n")

    # Prepare data for AIF360 formatting
    train, test = train_test_split(data, test_size=0.3, stratify=data['income'])

    train = StandardDataset(train, label_name='income', favorable_classes=[1],
                            protected_attribute_names=[protected_attr],
                            privileged_classes=[[1]]) # assuming 1 is the privileged class
    test = StandardDataset(test, label_name='income', favorable_classes=[1],
                           protected_attribute_names=[protected_attr],
                           privileged_classes=[[1]])

    # Scale data
    scaler = MinMaxScaler(copy=False)
    train.features = scaler.fit_transform(train.features)
    test.features = scaler.transform(test.features)

    # Define models
    def create_nn_model(input_dim):
        model = Sequential()
        model.add(Dense(64, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        return model

    # Stages of ROC process
    roc_stages = ["Pre", "Post"]

    print(f"Processing model: {selected_model_name}")\

    # Iterate for each DIR repair level
    for i, di_repair_level in enumerate(tqdm(np.linspace(0., 1., 11))):
        # First run DIR on data
        di = DisparateImpactRemover(repair_level=di_repair_level)
        if di_repair_level == 0:
            train_repd = train
            test_repd = test
        else:
            train_repd = di.fit_transform(train)
            test_repd = di.fit_transform(test)

        index = train.feature_names.index(protected_attr)
        X_tr = np.delete(train_repd.features, index, axis=1)
        X_te = np.delete(test_repd.features, index, axis=1)
        y_tr = train_repd.labels.ravel()

        # Train selected model
        if selected_model_name == "Neural Network":
            model = create_nn_model(X_tr.shape[1])
            model.fit(X_tr, y_tr, epochs=50, batch_size=32, verbose=0)
            y_pred_proba = model.predict(X_te)
            y_pred = (y_pred_proba > 0.5).astype(int).ravel()
        else:
            model = models[selected_model_name]
            model.fit(X_tr, y_tr)
            y_pred_proba = model.predict_proba(X_te)[:, 1].reshape(-1, 1)  # for ROC
            y_pred = model.predict(X_te)  # for DIR & ROC

        # Remaining data for running DIR & ROC
        test_repd_pred = test_repd.copy()
        test_repd_pred.scores = y_pred_proba  # for ROC
        test_repd_pred.labels = y_pred  # for DIR & ROC

        # Giving protected attribute binary notation
        p = [{protected_attr: 1}]
        u = [{protected_attr: 0}]

        # THE FOLLOWING METRICS ARE COMPUTED FOR THE MODEL PRE-ROC

        # Build object to compute metrics
        cls_metric = ClassificationMetric(test_repd, test_repd_pred, privileged_groups=p, unprivileged_groups=u)

        # Print & write other metrics before/mid/after processing
        if di_repair_level == 0:
            f.write("\nMetrics pre-ROC with NO pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)
        elif di_repair_level == 0.5:
            f.write("\nMetrics pre-ROC AFTER half-strength pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)
        elif di_repair_level == 1.0:
            # Compute other metrics after processing (max level)
            f.write("\nMetrics pre-ROC AFTER full-strength pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)

        # Save pre-ROC data to master dict
        save_metrics_to_dict(0, i, cls_metric)

        # Apply Reject Option Classification (ROC)
        roc = RejectOptionClassification(unprivileged_groups=u, privileged_groups=p, low_class_thresh=0.01,
                                         high_class_thresh=0.99, num_class_thresh=100, num_ROC_margin=50,
                                         metric_name=roc_metric, metric_ub=0.05, metric_lb=-0.05)
        roc = roc.fit(test_repd, test_repd_pred)
        test_repd_pred_roc = roc.predict(test_repd_pred)

        # THE FOLLOWING METRICS ARE COMPUTED FOR THE MODEL POST-ROC

        # Print & write other metrics before/mid/after processing
        if di_repair_level == 0:
            f.write("\nMetrics post-ROC with NO pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)
        elif di_repair_level == 0.5:
            f.write("\nMetrics post-ROC AFTER half-strength pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)
        elif di_repair_level == 1.0:
            # Compute other metrics after processing (max level)
            f.write("\nMetrics post-ROC AFTER full-strength pre-processing:\n")
            compute_metrics_for_txt(f, cls_metric)

        # Save post-ROC data to master dict
        cls_metric = ClassificationMetric(test_repd, test_repd_pred_roc, privileged_groups=p, unprivileged_groups=u)
        save_metrics_to_dict(1, i, cls_metric)

    # Save master dict data to CSV
    write_metrics_to_csv(filename)


# **** BASE CODE STARTS HERE ****

data = load_data()

# Map of protected attributes to test
protected_attrs = ['sex', 'race']

# Process each protected attribute config of dataset
for i, prot_attr in enumerate(protected_attrs):
    print(f"\nPROCESSING WITH '{prot_attr}' AS SENSITIVE ATTRIBUTE")
    process_dataset_dir_roc(prot_attr, data)

# Indicate success!
print()
print("Operation successful")

