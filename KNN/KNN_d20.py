import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Load the datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Separate features and target variable
X_train = df_train.iloc[:, 1:-1].values  # Assuming the last column is the target and the first column is an ID
y_train = df_train['Class'].values
X_test = df_test.iloc[:, 1:-1].values
y_test = df_test['Class'].values

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Feature Selection with RFE
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=20, step=1)
X_train_rfe = selector.fit_transform(X_train, y_train)
X_test_rfe = selector.transform(X_test)

# KNN model
k = 5  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# Training the model with RFE selected features
knn.fit(X_train_rfe, y_train)

# Making predictions
preds = knn.predict(X_test_rfe)
preds_proba = knn.predict_proba(X_test_rfe)[:, 1]  # Probability estimates for the positive class

# Model Evaluation
print(classification_report(y_test, preds, digits=4))
roc_auc = roc_auc_score(y_test, preds_proba)
print(f'ROC AUC Score: {roc_auc}')
