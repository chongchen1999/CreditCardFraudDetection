import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Loading the datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Separating features and labels
X_train = df_train.iloc[:, 1:-1]  # Features: all columns except the first and the last
y_train = df_train['Class']  # Labels: 'Class' column
X_test = df_test.iloc[:, 1:-1]  # Features: same as for training
y_test = df_test['Class']  # Labels: 'Class' column

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the GBDT classifier with specified parameters
model = GradientBoostingClassifier(
    n_estimators=2,
    learning_rate=0.1,
    max_depth=6,
    max_features='sqrt',
    subsample=0.8,
    random_state=10,
    min_impurity_decrease=0
)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Print the classification report
print(classification_report(y_test, y_pred, digits=4))
