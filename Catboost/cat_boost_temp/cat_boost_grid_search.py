import pandas as pd
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

# Load your dataset
df_train = pd.read_csv(r'/dataset/train_dataset_1to100.csv')
df_test = pd.read_csv(r'/dataset/test_dataset_1to100.csv')

"""train_data_0 = df_train.loc[df_train['Class'] == 0]
train_data_1 = df_train.loc[df_train['Class'] == 1]
auxiliar = train_data_1
for i in range(100):
    auxiliar=pd.concat([auxiliar, train_data_1])
df_train = pd.concat([train_data_0, auxiliar])"""

X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

# Initialize CatBoostClassifier with parameters that won't be tuned
model = CatBoostClassifier(task_type='GPU', eval_metric='AUC', devices='0:1', verbose=50)

# Define the parameter grid to search
param_grid = {
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [10, 15, 20],
    'iterations': [200, 300, 400],
    'bagging_temperature': [0.0, 0.2, 0.5]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='roc_auc', verbose=3)

# Fit the grid search
grid_search.fit(X_train, y_train)

# Print best parameters and best score
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best ROC AUC score: {grid_search.best_score_}")

# Evaluate on the test set with the best found model
best_model = grid_search.best_estimator_
preds = best_model.predict(X_test)
preds_proba = best_model.predict_proba(X_test)[:, 1]

# Compute and print classification report and ROC AUC score
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, preds, digits=4))
roc_auc = roc_auc_score(y_test, preds_proba)
print(f'ROC AUC Score: {roc_auc}')
