import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split

df_train = pd.read_csv(r'/dataset/train_dataset_1to100.csv')
df_test = pd.read_csv(r'/dataset/test_dataset_1to100.csv')

X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

X_sample, _, y_sample, _ = train_test_split(X_train, y_train, test_size=0.9, stratify=y_train, random_state=42)

# Initialize the CatBoostClassifier with specific parameters
catboost_model = CatBoostClassifier(eval_metric='AUC',
                                    task_type='GPU',
                                    devices='0:1',
                                    random_seed=1020,
                                    bagging_temperature=0.5,
                                    od_type='Iter',
                                    metric_period=50,
                                    od_wait=100,
                                    learning_rate=0.05,
                                    max_depth=10,
                                    iterations=400,
                                    verbose=50)

# Initialize RFE with CatBoostClassifier as the estimator
# You can specify the number of features to select with the n_features_to_select parameter
rfe = RFE(estimator=catboost_model, n_features_to_select=20, step=1)

# Fit RFE
rfe.fit(X_sample, y_sample)

# Transform training and test sets
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# Fit the model on the reduced dataset
# Note: We're fitting the CatBoostClassifier again because RFE doesn't automatically refit the final model
catboost_model.fit(X_train_rfe, y_train)  # Set verbose to 0 to reduce log clutter

# Make predictions
preds = catboost_model.predict(X_test_rfe)
preds_proba = catboost_model.predict_proba(X_test_rfe)[:, 1]

# Classification report and ROC AUC Score
print(classification_report(y_test, preds, digits=4))
roc_auc = roc_auc_score(y_test, preds_proba)


