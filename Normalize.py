import pandas as pd

df = pd.read_csv('merged_all.csv')
# Fill null values with 0
df = df.fillna(0)

# List of columns to normalize
columns_to_normalize = ['attachments', 'size', 'email_count_within_domain', 'total_recipients', 'total_external',
                        'disgruntled_person_email_count', 'outside_office_hours_email_count', 'cloud_count',
                        'job_ad_count', 'keylogger_count', 'exe_count', 'exe_count_outside_8_17', 'connect_count', 
                        'connect_outside_8_17', 'disconnect_outside_8_17', 'min_diff_office_start',
                        'max_diff_office_end', 'count_value_1', 'unique_pcs', 'count_value_1_outside_office_hours',
                        'distinct_systems_outside_office_hours', 'total_time_diff_x', 'total_time_diff_y',
                        'count_outside_office_hours', 'average_difference','entry_count']

# Convert negative values to positive
for column in columns_to_normalize:
    df[column] = df[column].abs()

# Normalize the columns
for column in columns_to_normalize:
    df[column] = 0 + ( (df[column] - df[column].min()) * (255 - 0) ) / (df[column].max() - df[column].min())

# Save the DataFrame back to csv
df.to_csv('merged_all_normalized.csv', index=False)