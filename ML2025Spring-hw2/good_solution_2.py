import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

# Load data
train_df = pd.read_csv('/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/train.csv')
test_df = pd.read_csv('/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/test.csv')

# Explore data
print(train_df.info())
print(train_df.describe())

# Handle missing values
imputer = SimpleImputer(strategy='mean')

# Feature selection
features = [
    'cli_day1', 'ili_day1', 'wnohh_cmnty_cli_day1', 'wbelief_masking_effective_day1',
    'wbelief_distancing_effective_day1', 'wcovid_vaccinated_friends_day1',
    'wlarge_event_indoors_day1', 'wothers_masked_public_day1', 'wothers_distanced_public_day1',
    'wshop_indoors_day1', 'wrestaurant_indoors_day1', 'wworried_catch_covid_day1',
    'hh_cmnty_cli_day1', 'nohh_cmnty_cli_day1', 'wearing_mask_7d_day1', 'public_transit_day1',
    'worried_finances_day1', 'tested_positive_day1',
    'cli_day2', 'ili_day2', 'wnohh_cmnty_cli_day2', 'wbelief_masking_effective_day2',
    'wbelief_distancing_effective_day2', 'wcovid_vaccinated_friends_day2',
    'wlarge_event_indoors_day2', 'wothers_masked_public_day2', 'wothers_distanced_public_day2',
    'wshop_indoors_day2', 'wrestaurant_indoors_day2', 'wworried_catch_covid_day2',
    'hh_cmnty_cli_day2', 'nohh_cmnty_cli_day2', 'wearing_mask_7d_day2', 'public_transit_day2',
    'worried_finances_day2', 'tested_positive_day2'
]

X = train_df[features]
y = train_df['tested_positive_day3']

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Model selection
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=KFold(n_splits=5), scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(pipeline.fit_transform(X_train), y_train)

best_model = grid_search.best_estimator_

# Evaluate model
y_pred_val = best_model.predict(pipeline.transform(X_val))
mse = mean_squared_error(y_val, y_pred_val)
print(f'Validation MSE: {mse}')

# Predict on test set
X_test = test_df[features]
test_predictions = best_model.predict(pipeline.transform(X_test))

# Save predictions
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': test_predictions
})

submission_df.to_csv('/content/submission.csv', index=False)