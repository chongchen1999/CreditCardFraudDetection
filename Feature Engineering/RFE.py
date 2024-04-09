from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

# Load your dataset
data = df

# Separate features and the target variable
X = data.drop(['id', 'Class'], axis=1)
y = data['Class']

# Initialize the Logistic Regression model
logistic_regression_model = LogisticRegression(max_iter=1000)

# X_small, _, y_small, _ = train_test_split(X, y, test_size=0.6, stratify=y, random_state=42)

# Variable to track the best number of features and corresponding accuracy
best_num_features = 0
best_accuracy = 0
accuracies = []

# Split the original dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Iterate over all possible numbers of selected features
for n_features_to_select in range(1, len(X.columns) + 1):
    selector = RFE(estimator=logistic_regression_model, n_features_to_select=n_features_to_select, step=1)
    selector.fit(X, y)
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    logistic_regression_model.fit(X_train_selected, y_train)
    y_pred = logistic_regression_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_num_features = n_features_to_select

# Print accuracies for each number of features
for i, accuracy in enumerate(accuracies, 1):
    print(f"Number of features: {i}, Accuracy: {accuracy:.12f}")

# Print the best number of features
print(f"\nThe best number of selected features is {best_num_features} with an accuracy of {best_accuracy:.12f}.")

selector = RFE(estimator=logistic_regression_model, n_features_to_select=best_num_features, step=1)
selector.fit(X, y)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)
logistic_regression_model.fit(X_train_selected, y_train)
y_pred = logistic_regression_model.predict(X_test_selected)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))