import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Prepare the data
X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

# Initialize the Random Forest classifier with specified parameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_features=None,
    random_state=None,
    max_leaf_nodes=None,
    min_impurity_decrease=0
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred, digits=4))
