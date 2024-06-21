import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# ************************************************ START CUSTOMIZATION ************************************************

# FULL NAME OF DATASET FILE TO PRE-PROCESS
dataset_filename = "adult.data"

# NEW FILE NAME FOR PRE-PROCESSED DATASET COPY (exclude .csv extension)
processed_dataset_filename = "adult.data_cleaned"

# ************************************************ END CUSTOMIZATION ************************************************


def cleanup_dataset(data):
    """
    Cleans up the provided dataset as you wish! (currently set up for Adult dataset)
    :param data: the pandas dataframe you wish to clean
    :return: the cleaned pandas dataframe
    """
    # Assign column names based on Adult dataset description
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                    'marital-status', 'occupation', 'relationship', 'race',
                    'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                    'native-country', 'income']
    data.columns = column_names

    # Remove column "education" as it is redundant with "education_num"
    data = data.drop(columns=['education'])

    # Encode categorical features using label encoding
    categorical_features = ['workclass', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'native-country']
    for feature in categorical_features:
        le = LabelEncoder()
        data[feature] = le.fit_transform(data[feature])

    # Remove duplicate entries
    data = data.drop_duplicates()

    # Remove rows with missing values
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    # Encode the target variable ('income')
    data['income'] = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

    return data


def load_and_prepare_data(dataset_file_name):
    """
    Loads the dataset from provided CSV file & cleans it according to the function defined above.
    :return: the pre-processed dataset as a pandas dataframe
    """
    print()
    print(f"Cleaning up dataset in {dataset_file_name}")

    # Load the data
    data = pd.read_csv(f',datasets/{dataset_file_name}', header=None)
    data_entry_ct_pre = len(data)

    cleaned_data = cleanup_dataset(data)

    # Prep strings for printing & writing
    entry_ct_pre_str = f"\nEntry count before cleanup: {data_entry_ct_pre}"
    entry_ct_post_str = f"Entry count after cleanup: {len(cleaned_data)}"
    entries_removed_str = f"Entries removed by pre-processing: {data_entry_ct_pre - len(cleaned_data)}"

    # Print dataset stats
    print(entry_ct_pre_str)
    print(entry_ct_post_str)
    print(entries_removed_str)
    print()
    print(f"Dataset successfully cleaned up and saved as {processed_dataset_filename}.csv")

    # Save the preprocessed dataset to a new CSV file
    cleaned_data.to_csv(f'cleaned-datasets/{processed_dataset_filename}.csv', index=False)


# **** BASE CODE STARTS HERE ****

load_and_prepare_data(f"{dataset_filename}")
