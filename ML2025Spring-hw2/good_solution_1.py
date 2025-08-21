import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
train_path = "/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/train.csv"
test_path = "/home/hllqk/projects/ML2025spring-hw/ML2025Spring-hw2/ML2025Spring-hw2-public/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Preprocessing
# Drop the 'id' column as it's not needed for prediction
X_train = train_df.drop(columns=['id', 'tested_positive_day3'])
y_train = train_df['tested_positive_day3']

# Select numerical and categorical columns
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

# Preprocessing for numerical data: Impute missing values
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data: Impute missing values and apply one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Create a pipeline that combines the preprocessor and the model
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Split the training data into training and validation sets
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train the model
pipeline.fit(X_train_split, y_train_split)

# Evaluate the model on the validation set
y_pred_val = pipeline.predict(X_val_split)
mse_val = mean_squared_error(y_val_split, y_pred_val)
print(f"Validation MSE: {mse_val}")

# Make predictions on the test set
X_test = test_df.drop(columns=['id'])
y_pred_test = pipeline.predict(X_test)

# Prepare the submission file
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'tested_positive_day3': y_pred_test
})

# Save the predictions to a CSV file
submission_df.to_csv('./content/submission.csv', index=False)