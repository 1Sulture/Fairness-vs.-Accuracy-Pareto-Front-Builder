import matplotlib.pyplot as plt
import pandas as pd

# ************************************************ START CUSTOMIZATION ************************************************

# DATASET'S FILE NAME TO READ FROM (excluding .csv extension)
dataset_file_name = "adult.data_cleaned"

# WHETHER TO PLOT GRAPH WITH GENDER & SIMPLIFIED RACE DATA
plot_general_graph = True

# WHETHER TO PLOT EXTRA RACE DATA
plot_bigger_race_graph = True

# ************************************************ END CUSTOMIZATION ************************************************

# Global lists for ordering graph data
gender_order = ["Male", "Female"]
race_order_binary = ['White', 'Other']
race_order_full = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']


def load_data():
    """
    Loads the dataset from provided CSV file & prints its entry count.
    :return: the dataset as a pandas dataframe
    """
    # Load the data
    data = pd.read_csv(f'cleaned-datasets/{dataset_file_name}.csv')

    # Prep strings for printing & writing
    dataset_file_str = f"Imported file name: {dataset_file_name}.csv"
    entry_ct_pre_str = f"Data entry count: {len(data)}\n"

    # Print dataset stats & write to file
    print()
    print(dataset_file_str)
    print(entry_ct_pre_str)
    print()

    return data


# Function for plotting general gender & race data
def plot_general_graph_func(dataset):
    # Simplified race mapping
    race_mapping = {1.0: 'White', 0.0: 'Other'}
    dataset['race'] = dataset['race'].map(race_mapping)

    # Gender Proportions
    gender_counts = dataset['sex'].value_counts(normalize=True).reindex(gender_order, fill_value=0) * 100
    gender_acceptance = dataset.groupby(['sex', 'income']).size().unstack().reindex(gender_order).fillna(0)
    gender_acceptance_percent = gender_acceptance.div(gender_acceptance.sum(axis=1), axis=0) * 100

    # Race Proportions
    race_counts = dataset['race'].value_counts(normalize=True).reindex(race_order_binary, fill_value=0) * 100
    race_acceptance = dataset.groupby(['race', 'income']).size().unstack().reindex(race_order_binary).fillna(0)
    race_acceptance_percent = race_acceptance.div(race_acceptance.sum(axis=1), axis=0) * 100

    # Plotting
    plt.rc('xtick', labelsize=13)
    plt.rc('ytick', labelsize=13)
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))


    # Gender Proportions
    axes[0, 0].bar(gender_counts.index, gender_counts.values, color=['blue', 'pink'])
    axes[0, 0].set_title('Gender Proportions', fontsize=14)
    axes[0, 0].set_ylabel('Percentage', fontsize=12)
    axes[0, 0].set_ylim([0, 80])
    for i, v in enumerate(gender_counts.values):
        axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Gender acceptance/rejection with percentages
    gender_acceptance.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'], ax=axes[0, 1])
    axes[0, 1].set_title('Gender Acceptance Rates', fontsize=14)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_xticklabels(gender_acceptance.index, rotation=0)  # set x-axis labels
    axes[0, 1].legend(fontsize=11)

    # Annotate with percentages
    for i, gender in enumerate(gender_acceptance.index):
        if 'Accepted' in gender_acceptance.columns:  # Check if the 'Accepted' category exists
            count_accepted = gender_acceptance.loc[gender, 'Rejected']
            percent_accepted = gender_acceptance_percent.loc[gender, 'Accepted']
            # Position the annotation at the top of the 'Accepted' bar
            axes[0, 1].text(i, count_accepted, f"{percent_accepted:.1f}%", ha='center', va='top', fontweight='bold',
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.6))

    # Race Proportions
    axes[1, 0].bar(race_counts.index, race_counts.values, color=['blue', 'orange'])
    axes[1, 0].set_title('Race Proportions', fontsize=14)
    axes[1, 0].set_ylabel('Percentage', fontsize=12)
    axes[1, 0].set_ylim([0, 90])
    for i, v in enumerate(race_counts.values):
        axes[1, 0].text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Race acceptance/rejection with percentages
    race_acceptance.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'], ax=axes[1, 1])
    axes[1, 1].set_title('Race Acceptance Rates', fontsize=14)
    axes[1, 1].set_ylabel('Count', fontsize=12)
    axes[1, 1].set_xticklabels(race_acceptance.index, rotation=0)  # set x-axis labels
    axes[1, 1].legend(fontsize=11)

    # Annotate with percentages
    for i, race in enumerate(race_acceptance.index):
        if 'Accepted' in race_acceptance.columns:
            count_accepted = race_acceptance.loc[race, 'Rejected']
            percent_accepted = race_acceptance_percent.loc[race, 'Accepted']
            # Position the annotation at the top of the 'Accepted' bar
            axes[1, 1].text(i, count_accepted, f"{percent_accepted:.1f}%", ha='center', va='top', fontweight='bold',
                            fontsize=12,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.6))

    plt.tight_layout()
    plt.savefig('generated-pics/adult_stats_general.png')


# Function to plot graph with details for all races
def plot_bigger_race_graph_func(dataset):
    # Map race to labels; revised race mapping with comprehensive labels
    race_mapping = {
        0: 'Amer-Indian-Eskimo',
        1: 'Asian-Pac-Islander',
        2: 'Black',
        3: 'Other',
        4: 'White'
    }

    # Apply the revised mapping to the dataframe
    dataset['race'] = dataset['race'].map(race_mapping)

    # Race proportions
    race_counts = dataset['race'].value_counts(normalize=True).reindex(race_order_full, fill_value=0) * 100
    race_acceptance = dataset.groupby(['race', 'income']).size().unstack().reindex(race_order_full).fillna(0)
    race_acceptance_percent = race_acceptance.div(race_acceptance.sum(axis=1), axis=0) * 100

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7))

    # Race proportions
    axes[0].bar(race_counts.index, race_counts.values, color=['blue', 'orange', 'green', 'red', 'purple'])
    axes[0].set_title('Race Proportions', fontsize=15)
    axes[0].set_ylabel('Percentage', fontsize=12)
    axes[0].set_ylim([0, max(race_counts.values) + 10])  # Adjust y-limit to fit all text labels
    axes[0].tick_params(axis='x', labelrotation=45, labelsize=10)
    axes[0].tick_params(axis='y', labelsize=12)
    for i, v in enumerate(race_counts.values):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Race acceptance/rejection with percentages
    race_acceptance.plot(kind='bar', stacked=True, color=['lightgreen', 'salmon'], ax=axes[1])
    axes[1].set_title('Race Acceptance Rates', fontsize=15)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].tick_params(axis='x', labelrotation=45, labelsize=10)
    axes[1].tick_params(axis='y', labelsize=12)
    axes[1].legend(fontsize=13)

    # Annotate with acceptance percentages only
    for i, race in enumerate(race_order_full):
        if 'Accepted' in race_acceptance.columns:
            count_accepted = race_acceptance.loc[race, 'Rejected']
            percent_accepted = race_acceptance_percent.loc[race, 'Accepted']
            axes[1].text(i, count_accepted, f"{percent_accepted:.1f}%",
                         ha='center', va='bottom', fontweight='bold', fontsize=12,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor='none', alpha=0.6))

    plt.tight_layout()
    plt.savefig('generated-pics/adult_stats_races.png')


# Load dataset
df = load_data()

# Mapping the labels
df['income'] = df['income'].apply(lambda x: 'Accepted' if x == 1 else 'Rejected')

# Map gender to labels
df['sex'] = df['sex'].map({1.0: 'Male', 0.0: 'Female'})

# Print respective graphs if requested
if plot_general_graph:
    print("Plotting graph with general data")
    plot_general_graph_func(df.copy())
    print("Successfully plotted")

print()

if plot_bigger_race_graph:
    print("Plotting graph with extra race data")
    plot_bigger_race_graph_func(df.copy())
    print("Successfully plotted")





