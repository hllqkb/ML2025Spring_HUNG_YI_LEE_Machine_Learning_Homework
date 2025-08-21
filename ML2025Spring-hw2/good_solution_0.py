# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
train_df = pd.read_csv('/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/train.csv')
test_df = pd.read_csv('/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/test.csv')

# Define the features and target
X_train = train_df.drop(['id', 'tested_positive_day3'], axis=1)
y_train = train_df['tested_positive_day3']
X_test = test_df.drop(['id'], axis=1)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the validation set
y_pred_val = model.predict(X_val)

# Evaluate the model on the validation set
mse = mean_squared_error(y_val, y_pred_val)
print(f'MSE on validation set: {mse}')

# Make predictions on the test set
y_pred_test = model.predict(X_test_scaled)

# Save the predictions to a submission.csv file
submission_df = pd.DataFrame({'id': test_df['id'], 'tested_positive_day3': y_pred_test})
submission_df.to_csv('/content/submission.csv', index=False)