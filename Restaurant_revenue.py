# -*- coding: utf-8 -*-

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "Restaurant_revenue (1).csv")

df = pd.read_csv(csv_path)

display(df.head())

df = pd.read_csv('/content/Restaurant_revenue (1).csv')
display(df.head())

"""## 1. Data Preprocessing and Initial Exploration"""

# Display basic information about the DataFrame, including data types and non-null values
df.info()

# Check for missing values in each column
print('\nMissing values per column:')
print(df.isnull().sum())

"""The `Cuisine_Type` column is categorical and needs to be converted into numerical format using one-hot encoding for the models. Also, it appears there are no missing values, which simplifies the preprocessing."""

# Apply one-hot encoding to the 'Cuisine_Type' column
df_processed = pd.get_dummies(df, columns=['Cuisine_Type'], drop_first=True)

# Display the first few rows of the processed DataFrame to see the new columns
print('DataFrame after one-hot encoding:')
display(df_processed.head())

"""## 2. Building a Linear Regression Model

For linear regression, we will predict `Monthly_Revenue`, which is a continuous numerical variable. All other relevant columns will be used as features.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Define features (X) and target (y) for Linear Regression
X_lin = df_processed.drop('Monthly_Revenue', axis=1)
y_lin = df_processed['Monthly_Revenue']

# Split the data into training and testing sets
X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train_lin, y_train_lin)

# Make predictions on the test set
y_pred_lin = linear_model.predict(X_test_lin)

# Evaluate the Linear Regression model
mae_lin = mean_absolute_error(y_test_lin, y_pred_lin)
mse_lin = mean_squared_error(y_test_lin, y_pred_lin)
rmse_lin = mse_lin**0.5
r2_lin = r2_score(y_test_lin, y_pred_lin)

print(f'Linear Regression Model Evaluation:')
print(f'Mean Absolute Error (MAE): {mae_lin:.2f}')
print(f'Mean Squared Error (MSE): {mse_lin:.2f}')
print(f'Root Mean Squared Error (RMSE): {rmse_lin:.2f}')
print(f'R-squared (R2): {r2_lin:.2f}')

"""## 3. Building a Logistic Regression Model

For logistic regression, we need a binary target variable. I will create a new target column, `High_Revenue`, which will be `1` if `Monthly_Revenue` is above its median, and `0` otherwise. This transforms the problem into a classification task (predicting whether revenue is high or low).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create a binary target variable for Logistic Regression
median_revenue = df_processed['Monthly_Revenue'].median()
df_processed['High_Revenue'] = (df_processed['Monthly_Revenue'] > median_revenue).astype(int)

# Define features (X) and new binary target (y) for Logistic Regression
X_log = df_processed.drop(['Monthly_Revenue', 'High_Revenue'], axis=1)
y_log = df_processed['High_Revenue']

# Split the data into training and testing sets
X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
logistic_model.fit(X_train_log, y_train_log)

# Make predictions on the test set
y_pred_log = logistic_model.predict(X_test_log)

# Evaluate the Logistic Regression model
accuracy_log = accuracy_score(y_test_log, y_pred_log)
conf_matrix_log = confusion_matrix(y_test_log, y_pred_log)
class_report_log = classification_report(y_test_log, y_pred_log)

print(f'Logistic Regression Model Evaluation:')
print(f'Accuracy: {accuracy_log:.2f}')
print(f'\nConfusion Matrix:\n{conf_matrix_log}')
print(f'\nClassification Report:\n{class_report_log}')

"""## 4. Deploying Predictive Tools with `joblib`

Now, I will save both the trained Linear Regression and Logistic Regression models using `joblib`. This allows you to load these models later for making new predictions without retraining them, which is essential for deployment.
"""

import joblib

# Save the Linear Regression model
joblib.dump(linear_model, 'linear_regression_model.joblib')
print('Linear Regression model saved as linear_regression_model.joblib')

# Save the Logistic Regression model
joblib.dump(logistic_model, 'logistic_regression_model.joblib')
print('Logistic Regression model saved as logistic_regression_model.joblib')

# To load a model later, you would use:
# loaded_linear_model = joblib.load('linear_regression_model.joblib')
# loaded_logistic_model = joblib.load('logistic_regression_model.joblib')

"""## App Interface Theme Suggestions for Fast Food Chains

Regarding the app interface themes related to fast-food chains, here are some ideas to make it attractive:

*   **Branding & Color Schemes:** Use vibrant and appealing color palettes typically associated with fast food (e.g., reds, yellows, oranges) or specific cuisine types (e.g., green for fresh, brown for coffee/burgers).
*   **Iconography:** Employ clear and intuitive icons for menu items, customer counts, marketing spend, etc. You could use burger, pizza, sushi icons for cuisine types.
*   **Layout & Responsiveness:** A clean, easy-to-navigate layout that is responsive across different devices (mobile, tablet, desktop) is crucial.
*   **Data Visualization:** Incorporate engaging charts and graphs to display predictions, model performance, and feature importance. For example, a bar chart showing predicted monthly revenue, or a gauge showing the probability of 'high revenue'.
*   **Interactive Elements:** Allow users to input hypothetical values for features (e.g., number of customers, menu price, marketing spend) and see real-time predictions.
*   **Fast Food Imagery:** Use high-quality images of popular fast-food items (burgers, fries, shakes) as background elements or in sections to enhance visual appeal.
*   **Dashboard Approach:** A dashboard-style interface for the predictive tool, showing key metrics and model outputs at a glance.
"""
