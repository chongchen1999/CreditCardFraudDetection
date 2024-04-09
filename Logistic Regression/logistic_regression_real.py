import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

def evaluate_model(X_train, X_test, y_train, y_test):
    """
    Trains a logistic regression model using the provided training data and evaluates it on the test data.

    Parameters:
    - X_train: Training feature data
    - X_test: Test feature data
    - y_train: Training target data
    - y_test: Test target data
    """
    # Initialize the Logistic Regression model
    model = LogisticRegression(C=1000, penalty='l1', solver='liblinear', max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.12f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

# Feature Selection with RFE
selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=29, step=1)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.fit_transform(X_test, y_test)

# Apply StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)
print("Evaluating model with StandardScaler:")
evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test)

# Apply MinMaxScaler
mm_scaler = MinMaxScaler()
X_train_mm_scaled = mm_scaler.fit_transform(X_train_selected)
X_test_mm_scaled = mm_scaler.transform(X_test_selected)
print("Evaluating model with MinMaxScaler:")
evaluate_model(X_train_mm_scaled, X_test_mm_scaled, y_train, y_test)