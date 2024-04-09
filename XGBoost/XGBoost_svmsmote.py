import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Loading the datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Separating features and labels
X_train = df_train.iloc[:, 1:-1]  # Features: all columns except the first and the last
y_train = df_train['Class']  # Labels: 'Class' column
X_test = df_test.iloc[:, 1:-1]  # Features: same as for training
y_test = df_test['Class']  # Labels: 'Class' column

from imblearn.over_sampling import SVMSMOTE

# Create an instance of SVMSMOTE
svmsmote = SVMSMOTE(sampling_strategy='auto', random_state=42)

# Fit the model and apply the oversampling
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)

# Check the new class distribution
print('Before SVMSMOTE:', y_train.value_counts())
print('After SVMSMOTE:', y_train_resampled.value_counts())

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Initialize the XGBoost classifier
model = XGBClassifier()

# Train the model
model.fit(X_train_scaled, y_train_resampled)

# Make predictions
preds = model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, preds, digits=4))


C_a = 50.0

y_test_flat = y_test.values
pred_flat = preds.flatten()
Amounts = df_test['Amount'].values  # Make sure this is a 1D array with the same length as y_test

Cost = np.sum(y_test_flat * (1 - pred_flat) * Amounts + C_a * pred_flat)
Cost_t = np.sum(y_test_flat * Amounts)

Savings = 1 - (Cost / Cost_t)
print(f"Savings: {Savings}")