import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_auc_score
from scipy.stats import mode

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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying KMeans
k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train_scaled)

# Assigning labels to clusters
clusters = kmeans.predict(X_train_scaled)
cluster_labels = {}
for i in range(k):
    cluster_labels[i] = mode(y_train[clusters == i]).mode[0]

# Predicting the Test set results
test_clusters = kmeans.predict(X_test_scaled)
test_preds = [cluster_labels[cluster] for cluster in test_clusters]

# Making the Confusion Matrix
print(classification_report(y_test, test_preds))