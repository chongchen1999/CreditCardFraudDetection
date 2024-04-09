import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier  # Changed to use GBDT
from sklearn.metrics import classification_report

# Loading the datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Separating features and labels
X_train = df_train.iloc[:, 1:-1]  # Features: all columns except the first and the last
y_train = df_train['Class']  # Labels: 'Class' column
X_test = df_test.iloc[:, 1:-1]  # Features: same as for training
y_test = df_test['Class']  # Labels: 'Class' column

# Feature scaling (While not required for tree-based models, maintaining for consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the GBDT classifier
model = GradientBoostingClassifier(n_estimators=100, random_state=42)  # Using Gradient Boosting

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred, digits=4))
