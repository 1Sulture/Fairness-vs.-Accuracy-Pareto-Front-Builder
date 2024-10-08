﻿# Fairness vs. Accuracy Pareto Front Builder

This repository contains all code files I used to generate my findings for my [bachelor thesis](https://repository.tudelft.nl/record/uuid:ee6bd5f1-8a87-480b-aa72-7e47c95b027f)!

## Brief Summary

This implementation allows for analyzing fairness and accuracy metric trends on a supplied dataset of choice. 
For fairness methods, it currently applies the *Disparate Impact Remover (DIR)* pre-processing and *Reject Option Classification (ROC)* post-processing techniques. 
The DIR is applied in 10 repair levels between 0 and 1, and the ROC either fully or not at all.

It can generate graphs showing either:
* How these metrics change across the DIR repair levels and when the ROC is applied.
* Pareto front pairs showing the best points based on the selected fairness vs. accuracy trade-off metrics and fairness weight.

## Content Details

All contained directories and code files are listed in chronological order.

Directories:
* `,datasets`: Put your raw dataset files here (`.data` type).
* `cleaned-datasets`: Cleaned datasets get saved here.
* `generated-csv-files`: Generated metric CSV files with all fairness and accuracy data from model runs go here.
* `generated-txt-files`: Generated TXT files corresponding to generated metric CSV files go here.
* `generated-pics`: Generated graphs go here.
* `sample-pics`: Contains sample graphs of model results analyzed in paper.

Code files:
* `dataset_cleaner`:
Cleans the dataset specified by its file name; must be located in `,datasets` directory.
* `roc-postprocess_log-reg_graphing`:
Runs *only* the ROC fairness technique on the specified cleaned dataset on *Logistic Regression* model, then plots its results.
* `dir-roc_all-models`:
Runs the DIR and ROC fairness techniques on the specified cleaned dataset & model, and saves all metric data to its respective CSV & TXT files.
* `data_visualizer`:
Generates graphs showing trends of selected metrics and Pareto front pair of specified metric pair.
* `adult_stats`:
Generates two graphs showing the gender and race statistics of the dataset.

## Guide to Analyzing the Included Adult Dataset

Note: All files have fields at the top that can be customized.

For a deeper overview of each code file's functionality, read the documentation and comments on the included functions.

1. **Preparing and cleansing the dataset**: Using the `dataset_cleaner.py` file.

    * **Operations**: Prepares and cleans your supplied dataset file.
    
    * **Usage**: At the top, fill in the full name of the dataset file to clean (note: currently the file reader is tuned for `.data` files, so you may need to tweak it for other file types) and the file name for the cleaned dataset copy.
   
    * **Result**: Should save the cleaned dataset copy to the `cleaned-datasets` directory.

2. **Testing the ROC to find optimal guiding metric**: Using the `roc-postprocess_log-reg_graphing.py` file.

    * **Operations**: On your provided dataset, runs the ROC on its 3 supported guiding metrics and plots their results, showing the ROC's performance on all 3  metrics.

    * **Usage**: At the top, fill in the cleaned dataset file name to test, and adjust the other graph customizations as desired.

    * **Result**: Should show and save a graph with the ROC's performance, also saving a TXT file with some of the statistics. Both would be saved in the `generated-pics` and `generated-txt-files` directories, respectively. File name consists of cleaned dataset's name, ROC label, protected attribute, and *Logistic Regression* abbreviation.

3. **Testing the selected model with DIR and ROC applied**: Using the `dir-roc_all-models.py` file.

    * **Operations**: Runs the DIR and ROC on the specified dataset, then saves all metric values for each processing stage in a CSV file & summarized values in a TXT file.

    * **Usage**: At the top, fill in the dataset file name to test, the selected ROC metric, ML model, and *K-Nearest Neighbors* neighbor count (if applicable).

    * **Result**: Should save all data in a CSV file and a TXT file summarizing all metrics. Both would be saved in the `generated-pics` and `generated-txt-files` directories, respectively. File name consists of cleaned dataset's name, DIR-ROC label, selected ROC metric, protected attribute, and model.

4. **Generating metric trend graphs**: Using the `data_visualizer.py` file.

    * **Operations**: Using your provided dataset CSV file, generates and saves two graphs, showing your selected fairness and accuracy metrics respectively. 

    * **Usage**: At the top, fill in the dataset file name to graph, the respective protected attribute (for labeling only), set `generate_metric_graphs` to `True`, title customization, and which supported fairness and accuracy metrics to plot (note: right below the customization area you can see all supported metrics).

    * **Result**: Should save both graphs in the `generated-pics` directory. File name consists of the read CSV file's name, metric type (fairness or accuracy), and `data-graphs` label.

5. **Generating Pareto front pairs**: Also using the `data_visualizer.py` file.

    * **Operations**: Using your provided dataset CSV file, generates two Pareto fronts on selected fairness and accuracy metric pair. Each front shows the trade-off across the DIR repair levels, where the left is without the ROC applied and the right with.

    * **Usage**: At the top, fill in the dataset file name to graph, the respective protected attribute (for labeling only), set `generate_pareto_fronts` to `True`, title customization, which supported fairness and accuracy metric to build the front with, and fairness weight (to customize emphasis on fairness).

    * **Result**: Should save Pareto front pair in the `generated-pics` directory. File name consists of the read CSV file's name, `pareto-front` label, abbreviated fairness & accuracy metrics, and fairness weight.


## Guide to Extend This Implementation

Below are steps to allow the implementation to process other datasets, models, metrics, and pre & post processing techniques.


* **Using other datasets**: In the `dataset_cleaner.py` file.

    In the `cleanup_dataset` function: 

    1. Line 23: Update `column_names` array with all columns of dataset, in order.
    2. Line 29: Add/remove column drop commmands as desired.
    3. Line 33: Update `categorical_features` array with all non-numerical column names of dataset.
    4. Line 39: Add/remove any remaining cleaning steps as desired.
    5. Line 47: Update tested label column's values, ensuring its binary encoded.

    In the `load_and_prepare_data` function: 
        
    * Line 61: Only modify the `read_csv` function if reading file type is not `.data`.


* **Adding other models for DIR-ROC processing**: In the `dir-roc_all-models.py` file.

    Outside the functions:

    1. Line 34: Add comment with new model's name.
    2. Line 49: Add new model's full name to end of `model_names` array.
    3. Line 57: Add abbreviation of new model to end of `model_abbrev` array.

    In the `process_dataset_dir_roc` function:

    * Line 221: Add new model's full name (same as in `model_names`) & object instance to the `models` dict.

    Note: More complex models, such as *Neural Networks*, might need modifications beyond the steps given above.


* **Adding more metrics for `data-visualizer`'s graphs**: In two files.
    
    Note: This will modify the way the metric CSV file stores data.

    For the `dir-roc_all-models.py` file:

    * Outside the functions:

        * Line 63 or 65: Add full name of new metric to the end of the respective array.
    
    * In the `save_metrics_to_dict` function:

        * Line 114 or 125: For the respective Dict, add a reference to your new metric and set its value to the metric's computation (the `AIF360` library may help).

    * In the `compute_metrics_for_txt` function:

        * Line 190 or 202: Same as step above.


    For the `data-visualizer.py` file:
    
    * Outside the functions:

        * Line 61 or 63: Add full name of new metric to the end of the respective array.
        * Line 67 or 68: Add abbreviation of new metric to the end of the respective array.

    * In the `plot_selected_metric_data` function:

        * Line 171: Check if the included graph lines should be changed to match your new metric's values.
        
    * In the `plot_pareto_front_pair` function:

        * Line 347: If adding new accuracy metric, check if accuracy scoring needs adjustments. By default it assumes 1 as the best and 0 as the worst value.

        * Line 353: If adding new fairness metric, check if fairness scoring needs adjustments. By default it assumes 0 as the optimal value, with deviating values having a linear penalty.


    P.S. You can also update the `roc-postprocess_log-reg_graphing.py` with your new metrics if desired.


* **Using a different pre-processing technique**: In at least two files.

    Note: The following steps assume your new technique has at least some parameter to quantify its intensity. If not, then your loop, metric CSV writer & reader, and metric & Pareto front graphing functions will need further adjustments.

    For the `dir-roc_all-models.py` file:
    
    * In the `process_dataset_dir_roc` function:
    
        1. Line 267: Change the loop to iterate at each intensity level of your new technique.
        2. Line 269: Replace the `DisparateImpactRemover` object definition with the pre-processing object you choose to use.
        3. Line 274: Changes on how the train & test sets are processed may be necessary.
        4. Line 321 & 346: Changes on how the master Dict object holding all metric data stores its values may be necessary.
        5. Line 349: Changes on how the data is saved to the metric CSV file may be necessary.


    For the `data-visualizer.py` file: 
    
    * In the `plot_selected_metric_data` function:

        * Line 81: If CSV data structure is changed, then this function, data structures holding new data format, and dependent functions may need adustments.
        * Line 133: Update the x-axis label to that of your new technique.


* **Using a different post-processing technique**: In at least two files.

    Note: The following steps assume your new technique is either applied fully or not at all. If not, then your loop, metric CSV writer & reader, and metric & Pareto front graphing functions will need further adjustments.


    For the `dir-roc_all-models.py` file:
    
    * In the `process_dataset_dir_roc` function:
    
        1. Line 274: Changes on how the train & test sets are processed may be necessary.
        2. Line 324-328: Replace the `RejectOptionClassification` object definition with the post-processing object you choose to use.
        3. Line 321 & 346: Changes on how the master Dict object holding all metric data stores its values may be necessary.
        4. Line 349: Changes on how the data is saved to the metric CSV file may be necessary.
    

    For the `data-visualizer.py` file:
    
    * In the `plot_selected_metric_data` function:
    
        * Line 81: If CSV data structure is changed, then this function, data structures holding new data format, and dependent functions may need adustments.
        * Line 152: Update the loop and imported values, following new metric CSV file format if necessary.
    
    * In the `plot_pareto_front_pair` function:
    
        * Line 436: Functions used for generating Pareto fronts likely will need updates.

    P.S. You can also update the `roc-postprocess_log-reg_graphing.py` with your new technique if desired.



## Final Remarks

I hope you found this guide helpful and put this implementation to good use!

If you feel I missed explaining something or there's an issue/bug in the code, feel free to reach out. Feedback is always appreciated!

