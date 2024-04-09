import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import  PolynomialFeatures
from sklearn.model_selection import GridSearchCV

# Load your dataset
df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\reduced_data.csv')

# Separate features and target
X = df.iloc[:, 1:-1]  # Exclude 'id' and 'Class' columns
y = df['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Feature Selection with RFE
selector = RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=20, step=1)
X_selected = selector.fit_transform(X, y)

# Split the dataset into training and test sets
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.5, random_state=42)

# Apply PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_selected)
X_test_poly = poly.transform(X_test_selected)

print("Evaluating logic model with PolynomialFeatures:")

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'penalty': ['l1', 'l2'],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']
}

logistic = LogisticRegression(max_iter=1000)
clf = GridSearchCV(logistic, param_grid, cv=5)
clf.fit(X_train_poly, y_train)
print("Best parameters: ", clf.best_params_)

best_params = clf.best_params_
model = LogisticRegression(**best_params, max_iter=1000)

model.fit(X_train_poly, y_train)
prediction = model.predict(X_test_poly)
print(classification_report(y_test, prediction, digits=4))

# Output training results for each combination of hyperparameters
print("Training results for each hyperparameter combination:")
for i, params in enumerate(clf.cv_results_['params']):
    print(f"Combination {i+1}: {params}")
    for score in ['mean_test_score', 'std_test_score', 'rank_test_score']:
        print(f"  {score}: {clf.cv_results_[score][i]}")