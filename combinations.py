import pandas as pd
import numpy as np
from itertools import product

# Read your CSV file into a DataFrame
df = pd.read_csv('merged_all_normalized.csv')

# Assuming features is a list of your feature names
features = ['attachments', 'size', 'email_count_within_domain', 'total_recipients', 'total_external', 
            'disgruntled_person_email_count', 'outside_office_hours_email_count', 'cloud_count', 
            'job_ad_count', 'keylogger_count', 'exe_count', 'exe_count_outside_8_17', 'connect_count', 
            'connect_outside_8_17', 'disconnect_count', 'min_diff_office_start', 'max_diff_office_end', 
            'count_value_1', 'unique_pcs', 'count_value_1_outside_office_hours', 
            'distinct_systems_outside_office_hours', 'total_time_diff_x', 'total_time_diff_y', 
            'count_outside_office_hours', 'average_difference','entry_count']

# Define your thresholds here. Make sure the order of thresholds matches the order of features.
thresholds = list(range(71, 74))

# Create a dictionary to store the results for each set of thresholds
results = {}

# Generate all combinations of thresholds for each feature
threshold_sets = list(product(thresholds, repeat=len(features)))

# Try each set of thresholds
for threshold_set in threshold_sets:
    # Apply each threshold to the corresponding feature
    for feature, threshold in zip(features, threshold_set):
        df[feature] = np.where(df[feature] > threshold, 1, 0)

    # Label each row based on whether the majority of its features are above the threshold
    df['label'] = df[features].mean(axis=1) > 0.4

    # Convert the boolean labels to integers
    df['label'] = df['label'].astype(int)

    # Count the number of labels of 1's and 0's
    label_counts = df['label'].value_counts().to_dict()

    # Store the results for this set of thresholds
    results[threshold_set] = label_counts

# Convert the results to a DataFrame
results_df = pd.DataFrame(results).T

# Save the results to a CSV file
results_df.to_csv('threshold_results.csv')