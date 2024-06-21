# Fairness vs. Accuracy Pareto Front Builder

This repository contains all code files I used to generate my findings for my bachelor thesis!

## Brief Summary

This implementation allows for analyzing fairness and accuracy metric trends on a supplied dataset of choice. 
For fairness methods, it currently applies the *Disparate Impact Remover (DIR)* pre-processing and *Reject Option Classification (ROC)* post-processing techniques. 
The DIR is applied in 10 repair levels between 0 and 1, and the ROC either fully or not at all.

It can generate graphs showing either:
* How these metrics change across the DIR repair levels and when the ROC is applied.
* Pareto front pairs showing the best points based on the selected fairness vs. accuracy trade-off metrics.

## Content Details

All containing directories and files are listed in chronological order.

Directories:
* **,datasets**: Put your raw dataset files here.
* **cleaned-datasets**: Cleaned datasets get saved here.
* **generated-csv-files**: Generated CSV files with all fairness and accuracy data from model runs go here.
* **generated-txt-files**: Generated TXT files corresponding to generated CSV files go here.
* **generated-pics**: Generated graphs go here.
* **sample-pics**: Contains sample graphs of model results analyzed in paper.

Code files:
* **dataset_cleaner**:
Cleans the dataset specified by its file name; must be located in ",datasets" directory.
* **roc-postprocess_log-reg_graphing**:
Runs *only* the ROC fairness technique on the specified cleaned dataset on *Logistic Regression* model, then plots its results.
* **dir-roc_all-models**:
Runs the DIR and ROC fairness techniques on the specified cleaned dataset & model, and saves all metric data to its respective CSV & TXT files.
* **data_visualizer**:
Generates graphs showing trends of selected metrics and Pareto front pair of specified metric pair.
* **adult_stats**:
Generates two graphs showing the gender and race statistics of the dataset.

## Guide to Analyzing the Included Adult Dataset

Note that all files have fields at the top that can be customized.

For a deeper overview on each code file's functionality, read the docmentation and comments of the included functions.

1. **Preparing and cleansing the dataset**: Using the **dataset_cleaner.py** file.

    **Operations**: Prepares and cleans your supplied dataset file.
    
    **Instructions**: At the top, fill in the full name of the dataset file to clean (note: currently the file reader is tuned for .data files, so you may need to tweak it for other file types). Also fill in the file name for the cleaned dataset copy. 
   
    **Result**: Should save the cleaned dataset copy to the **cleaned-datasets** directory.

2. **Testing the ROC to find optimal guiding metric**: Using the **roc-postprocess_log-reg_graphing** file.

    **Operations**: On your provided dataset, runs the ROC on its 3 supported guiding metrics and plots their results, showing the ROC's performance on all 3 metrics.

    **Instructions**: At the top, fill the cleaned dataset file name to test, and adjust the other graph customizations as desired.

    **Result**: Should show and save a graph with the ROC's performance, also saving a TXT file with some of the statistics. Both would be saved in the **generated-pics** and **generated-txt-files** directories respectively.

3. **Testing the selected model with DIR and ROC applied**: Using the **dir-roc_all-models** file.

    **Operations**: Runs the DIR and ROC on the specified dataset, then saves all metric values for each processing stage in a CSV file.

    **Instructions**: At the top, fill the dataset file name to test, the selected ROC metric, ML model, and *K-Nearest Neighbors* neighbor count (if applicable).

    **Result**: Should save all data in a CSV file in the **generated-csv-files** directory, with the name consisting of the cleaned dataset's name and other markers indicating the selected roc metric, protected attribute, and model. Also saves a TXT file with the same name, summarizing all metrics.

4. **Generating metric trend graphs**: Using the **data_visualizer** file.

    **Operations**: On your provided dataset CSV file, generates and saves two graphs, showing your selected fairness and accuracy metrics respectively. 

    **Instructions**: At the top, fill the dataset file name to graph, the respective protected attribute (for labeling only), title customization, and which supported fairness and accuracy metrics to plot (note: right below the customization area you can see all supported metrics).

    **Result**: Should save both graphs in the **generated-pics** directory, with the name consisting of the CSV file's name and marker indicating whether fairness or accuracy is displayed. 

5. **Generating Pareto front pairs**: Also using the **data_visualizer** file.

    **Operations**: On your provided dataset CSV file, generates two Pareto fronts on both selected farness and accuracy metrics. Each front shows the trade-off across the DIR repair levels, where the left is without the ROC applied and the right with.

    **Instructions**: At the top, fill the dataset file name to graph, the respective protected attribute (for labeling only), title customization, which supported fairness and accuracy metric to build the front with, and fairness weight (to customize emphasis on fairness).

    **Result**: Should save Pareto front pair in the **generated-pics** directory, with the name consisting of the CSV file's name, the front's fairnes and accuracy metrics, and fairness weight.


<!-- ## Guide to Extend This Implementation -->

<!-- *coming soon...* -->