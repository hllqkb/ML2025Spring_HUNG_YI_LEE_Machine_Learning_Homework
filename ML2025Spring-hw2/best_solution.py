import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Define paths
data_dir = '/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public'
train_path = os.path.join(data_dir, 'train.csv')
test_path = os.path.join(data_dir, 'test.csv')
submission_path = './content/submission.csv'

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Explore data
print(train_df.head())
print(train_df.info())
print(train_df.describe())

# Handle missing values
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Feature selection
features = [col for col in train_df.columns if col not in ['id', 'tested_positive_day3']]
X = train_df[features]
y = train_df['tested_positive_day3']
X_test = test_df[features]

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Model selection and training
model = RandomForestRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model
best_model = grid_search.best_estimator_

# Evaluate model
y_pred_val = best_model.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_pred_val)
print(f'Validation MSE: {mse}')

# Predict test set
y_pred_test = best_model.predict(X_test_scaled)

# Prepare submission
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': y_pred_test
})

# Save predictions
submission_df.to_csv(submission_path, index=False)
print(f'Submissions saved to {submission_path}')