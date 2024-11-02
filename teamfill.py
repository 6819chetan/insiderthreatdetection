import pandas as pd

# Assuming df is your dataframe
df = pd.read_csv('2011-03.csv')

from sklearn.ensemble import RandomForestClassifier

# Example: Predict missing 'team' values
# Separate data with and without missing 'team'
train_df = df[df['team'].notna()]
test_df = df[df['team'].isna()]

# Features and target variable
features = ['department', 'business_unit', 'functional_unit', 'role', 'supervisor']

combined_df = pd.concat([train_df[features], test_df[features]])

# Create dummy variables
combined_df = pd.get_dummies(combined_df)

# Split the data back into training and test sets
X_train = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

y_train = train_df['team']

# Training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the missing values
df.loc[df['team'].isna(), 'team'] = model.predict(X_test)

# Repeat for 'department' and 'functional_unit'

# Save the updated dataframe
df.to_csv('2011-03.csv', index=False)
