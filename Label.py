import pandas as pd
import numpy as np

# Read your CSV file into a DataFrame
df = pd.read_csv('merged_all_normalized.csv')
df1 = pd.read_csv('merged_all_normalized.csv')

# Assuming features is a list of your feature names
features = ['attachments', 'size', 'email_count_within_domain', 'total_recipients', 'total_external', 
            'disgruntled_person_email_count', 'outside_office_hours_email_count', 'cloud_count',
            'job_ad_count', 'keylogger_count', 'exe_count', 'exe_count_outside_8_17', 'connect_count',
            'connect_outside_8_17', 'disconnect_outside_8_17', 'min_diff_office_start', 'max_diff_office_end',
            'count_value_1', 'unique_pcs', 'count_value_1_outside_office_hours', 
            'distinct_systems_outside_office_hours', 'total_time_diff_x', 'total_time_diff_y', 
            'count_outside_office_hours', 'average_difference','entry_count']

# Define your thresholds here. Make sure the order of thresholds matches the order of features.
# thresholds = [86.7,99.4,94.7,135.841,100.45,127.5,77.91,27.625,37.5,5.67,85,127.5,163.93,63.75,163.93,120.506,234.4117,141.667,127.5,95.625,109.285,114.3073,120.28,51,86.024,0] # 5000 instances - 1's count - 38
# thresholds = [71,88,72,114,92,113,63,21,27,0,85,127,145,63,145,119,226,85,42,63,72,39,0,51,81,0] # 10000 instances - 1's count - 196
# weightage = [0.33,0.33,0.33,0.33,0.33,0.33,0.33,  0.33,1,1,1,1,1,1,1,  0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66, 1]

# thresholds = [20.4,69.02,7,59,38,63,7,4,7,0,0,0,18,0,18,108,193,56,42,31,36,7,0,51,3,0]

# thresholds = [35,74,7,73,61,99,21,4,7,0,0,0,18,0,18,22,37,56,0,31,36,21,0,51,5,0] # 50000 instances - 1's count - 1522
# thresholds = [40.8,75.87,0,78.64,61.82,106.25,21,8,10,0,0,0,18,0,18,23,40,56,0,31,38,23,0,51,9,0] # 40000 instances 
# thresholds = [51,78,0,85,69,106,35,10,12,0,0,0,36,0,0,26,41,56,0,31,36,25,0,51,17,0] # 30000 instances
# thresholds = [56,81,21,97,77,106,49,14,17,0,0,0,54,0,0,28,44,56,42,31,36,29,0,51,27,0] # 20000 instances
# thresholds = [61,83,29,100,85,106,49,15,20,0,0,0,54,0,0,28,44,56,42,31,36,30,0,51,53,0] # 18000 instances
thresholds = [62,83.22,29.15,100.1,85,106.26,49.58,17,20,0,0,0,54,0,0,28,44,56,42,31,36,30,0,51,54,0] # 17500 instances
# thresholds = [61,83,29,100,85,106,49,17,20,0,0,0,72,0,72,28,44,56,42,31,36,30,0,51,54,0] # 17400 instances
# thresholds = [61,83,36,102,85,106,49,17,20,0,0,0,72,0,72,28,44,85,42,31,36,30,0,51,54,0] # 17000 instances


# thresholds = [71,88,72,114,92,113,63,21,27,0,85,127,145,63,145,119,226,85,42,63,72,39,0,51,81,0] # 10000 instances 

# thresholds = [35,74,7,73,61,  113  ,21,4,17,0,   80    , 0,18,0,18,22,37,  85   ,0,31,36,    39   ,0,51,5,0] # 50000 + 10000 instances - 1's count - 985



# # Apply threshold to each feature
# for feature, threshold in zip(features, thresholds):
#     df[feature] = np.where(df[feature] > threshold, 1, 0)

# # Label each row based on whether the majority of its features are above the threshold
# df['label'] = df[features].mean(axis=1) > 0.5

# # Convert the boolean labels to integers
# df['label'] = df['label'].astype(int)

# # Count the number of labels of 1's and 0's
# label_counts = df['label'].value_counts()

# # Save the resulting DataFrame to a new CSV file
# df.to_csv('result.csv', index=False)

# # Filter rows where label is 1
# label_1_df = df[df['label'] == 1]

# # Count the number of unique users
# unique_users = label_1_df['user'].nunique()

# # Print the number of unique users
# print(unique_users)

# # Print the counts of 1's and 0's
# print(label_counts)



# Define the weightage for each feature
weightage = [0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,0.33,1,1,1,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.66,0.33] 

# Apply threshold to each feature and multiply by its weight
for feature, threshold, weight in zip(features, thresholds, weightage):
    df[feature] = np.where(df[feature] > threshold, 1, 0) * weight

# Label each row based on whether the weighted mean of its features is above the threshold
df['label_weighted'] = df[features].sum(axis=1) / sum(weightage) > 0.5

# Convert the boolean labels to integers
df['label_weighted'] = df['label_weighted'].astype(int)

# Filter rows where label_weighted is 1
label_1_weighted_df = df[df['label_weighted'] == 1]

# Count the number of unique users
unique_users_weighted = label_1_weighted_df['user'].nunique()

print(df['label_weighted'].value_counts())

df1['label_weighted'] = df['label_weighted']

# Print the number of unique users
print(unique_users_weighted)

# # Save the resulting DataFrame to a new CSV file
df1.to_csv('merged_labeled.csv', index=False)




# 40000 + weighted - 1's count - 1661 , 1-users - 78
# 50000 + weighted - 1's count - 2743 , 1-users - 118