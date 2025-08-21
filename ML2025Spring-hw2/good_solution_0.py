import pandas as pd

# Define the paths
train_path = "/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/train.csv"
test_path = "/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/test.csv"

# Load the datasets
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Fill missing values with the median for numerical columns and mode for categorical columns
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)

# Drop the 'id' column and separate the target variable
X_train = train_df.drop(columns=['id', 'tested_positive_day3'])
y_train = train_df['tested_positive_day3']

# Drop the 'id' column for the test set
X_test = test_df.drop(columns=['id'])

from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Predict on the training set to evaluate
y_pred_train = model.predict(X_train)

# Calculate MSE
mse = mean_squared_error(y_train, y_pred_train)
print(f'Mean Squared Error: {mse}')

# Predict on the test set
y_pred_test = model.predict(X_test)

# Create a DataFrame for submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': y_pred_test
})

# Save the predictions to a CSV file
submission_df.to_csv('./content/submission.csv', index=False)