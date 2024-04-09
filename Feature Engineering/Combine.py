from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
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
    model = LogisticRegression(max_iter=1000)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.12f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))

# Load your dataset
data = df

# Separate features and the target variable
X = data.drop(['id', 'Class'], axis=1)
y = data['Class']

# Feature Selection with RFE
selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=20, step=1)
X_selected = selector.fit_transform(X, y)

# Split the dataset into training and test sets
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.5, random_state=42)

# Apply PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)
print("Evaluating model with PolynomialFeatures:")
evaluate_model(X_train_poly, X_test_poly, y_train, y_test)

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