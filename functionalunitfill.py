import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Assuming df is your dataframe
df = pd.read_csv('2011-03.csv')

# Features and target variable
features = ['team', 'business_unit', 'department', 'role', 'supervisor']

# Predict missing 'functional_unit' values
# Separate data with and without missing 'functional_unit'
train_df = df[df['functional_unit'].notna()]
test_df = df[df['functional_unit'].isna()]


combined_df = pd.concat([train_df[features], test_df[features]])
# Create dummy variables
combined_df = pd.get_dummies(combined_df)

# Split the data back into training and test sets
X_train = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

y_train = train_df['functional_unit']

# Training the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predicting the missing values
df.loc[df['functional_unit'].isna(), 'functional_unit'] = model.predict(X_test)

# Save the updated dataframe
df.to_csv('2011-03.csv', index=False)