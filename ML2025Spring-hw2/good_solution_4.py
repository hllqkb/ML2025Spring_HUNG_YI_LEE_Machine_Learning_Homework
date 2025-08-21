import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import os

# Set the directory path
data_dir = '/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public'

# Load datasets
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Explore the data
print(train_df.head())
print(train_df.info())
print(train_df.describe())

# Handle missing values
# Impute missing values with median for numerical features and most frequent for categorical features
numerical_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

# Remove 'id' and 'tested_positive_day3' from numerical features
numerical_features.remove('id')
numerical_features.remove('tested_positive_day3')

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Define the model pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))])

# Split the training data into training and validation sets
X_train = train_df.drop(columns=['id', 'tested_positive_day3'])
y_train = train_df['tested_positive_day3']
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [3, 5, 7],
    'regressor__learning_rate': [0.01, 0.1]
}

grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the model on the validation set
y_pred_val = best_model.predict(X_val)
mse_val = mean_squared_error(y_val, y_pred_val)
print(f"Validation MSE: {mse_val}")

# Make predictions on the test set
X_test = test_df.drop(columns=['id'])
predictions = best_model.predict(X_test)

# Prepare the submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': predictions
})

# Save predictions to the specified output file
submission_df.to_csv('/content/submission.csv', index=False)

print("Predictions saved to /content/submission.csv")