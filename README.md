# Fairness vs. Accuracy Pareto Front Builder

This repository contains all code files I used to generate my findings for my bachelor thesis!

## Brief Summary

This implementation allows for analyzing fairness and accuracy metric trends on a supplied dataset of choice. 
For fairness methods, it currently applies the *Disparate Impact Remover (DIR)* pre-processing and *Reject Option Classification (ROC)* post-processing techniques. 
The DIR is applied in 10 repair levels between 0 and 1, and the ROC either fully or not at all.
It can generate graphs showing either:
* How these metrics change across the DIR repair levels and whether the ROC is applied.
* Pareto front pairs showing the best points based on the selected fairness vs. accuracy trade-off metrics.

## Content details

Each directory and file is listed in chronological order.

We have the following directories:
* **,datasets**: Put your raw dataset files here.
* **cleaned-datasets**: Cleaned datasets get saved here.
* **generated-csv-files**: Generated CSV files with all fairness and accuracy data from model runs go here.
* **generated-txt-files**: Generated TXT files corresponding to generated CSV files go here.
* * **generated-pics**: Generated graphs go here.
* **sample-pics**: Sample graphs of model results analyzed in paper are here.

And the following code files:
* **dataset_cleaner**:
Cleans the dataset specified by its file name; must be located in ",datasets" directory.
* **dir-roc_all-models**:
Runs the DIR and ROC fairness techniques on the specified cleaned dataset & model, and saves all metric data to its respective CSV & TXT files.
* **roc-postprocess_log-reg_graphing**:
Runs *only* the ROC technique on the specified cleaned dataset on *Logistic Regression* model, then plots results.
* **data_visualizer**:
Generates graphs showing trends of selected metrics and Pareto front pair of specified metric pair.
* **adult_stats**:
Generates two graphs showing the gender and race statistics of the dataset.