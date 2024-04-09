import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  # Importing SVC for SVM
from sklearn.metrics import accuracy_score, classification_report

# Load datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Splitting the datasets into features and target variable
X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Creating the SVM model
svm_model = SVC(max_iter=10000)  # You can adjust other parameters as needed
svm_model.fit(X_train, y_train)  # Training the model
y_pred = svm_model.predict(X_test)  # Making predictions

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
