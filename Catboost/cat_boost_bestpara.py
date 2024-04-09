import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load your dataset
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Define features and target variable
X_train = df_train.iloc[:, 1:-1]  # Assuming the first column is an index or identifier and the last column is the target
y_train = df_train['Class']
X_test = df_test.iloc[:, 1:-1]
y_test = df_test['Class']

# Initialize CatBoostClassifier with the best parameters
model = CatBoostClassifier(
    task_type='GPU',
    eval_metric='AUC',
    devices='0:1',
    verbose=50,
    bagging_temperature=0.5,
    iterations=400,
    learning_rate=0.05,
    max_depth=10
)

# Fit the model
model.fit(X_train, y_train)

# Make predictions
preds = model.predict(X_test)

# Compute and print classification report and ROC AUC score
print(classification_report(y_test, preds, digits=4))

C_a = 50.0

y_test_flat = y_test.values
pred_flat = preds.flatten()
Amounts = df_test['Amount'].values  # Make sure this is a 1D array with the same length as y_test

Cost = np.sum(y_test_flat * (1 - pred_flat) * Amounts + C_a * pred_flat)
Cost_t = np.sum(y_test_flat * Amounts)

Savings = 1 - (Cost / Cost_t)
print(f"Savings: {Savings}")