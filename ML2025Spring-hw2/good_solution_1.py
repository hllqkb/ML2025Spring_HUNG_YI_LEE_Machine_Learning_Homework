import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

# Load data
data_dir = '/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public'
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Data Analysis
print(train_df.info())
print(train_df.describe())

# Feature Selection
# Drop columns not needed for prediction
features_to_drop = ['id', 'cli_day3', 'ili_day3', 'wnohh_cmnty_cli_day3', 'wbelief_masking_effective_day3',
                    'wbelief_distancing_effective_day3', 'wcovid_vaccinated_friends_day3', 'wlarge_event_indoors_day3',
                    'wothers_masked_public_day3', 'wothers_distanced_public_day3', 'wshop_indoors_day3',
                    'wrestaurant_indoors_day3', 'wworried_catch_covid_day3', 'hh_cmnty_cli_day3', 'nohh_cmnty_cli_day3',
                    'wearing_mask_7d_day3', 'public_transit_day3', 'worried_finances_day3']
X = train_df.drop(columns=features_to_drop + ['tested_positive_day3'])
y = train_df['tested_positive_day3']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Model Pipeline
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor(random_state=42))])

# Hyperparameter Tuning
param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
    'regressor__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluation
y_pred_val = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred_val)
print(f'Validation MSE: {mse}')

# Prediction on Test Set
X_test = test_df.drop(columns=['id'])
predictions = best_model.predict(X_test)

# Save Predictions
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': predictions
})

submission_df.to_csv('./content/submission.csv', index=False)