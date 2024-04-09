import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Loading datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Preparing the data
X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

# Instantiating and fitting the Decision Tree model with specified parameters
decision_tree_model = DecisionTreeClassifier(
    splitter="best",
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_leaf_nodes=None,
    min_impurity_decrease=0,
    max_features=None,
    random_state=None  # Changed 'randomstate' to 'random_state' to correct the parameter name
)

decision_tree_model.fit(X_train, y_train)

# Making predictions
y_pred = decision_tree_model.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
