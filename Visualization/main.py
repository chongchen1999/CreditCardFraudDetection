from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\creditcard_2023.csv')

# Assuming `df` is your DataFrame with the V1-V28 features and 'Class' as the target variable
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:, 'Class'].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)


principalComponents = pca.fit_transform(x)

# Creating a DataFrame with the principal components and the Class
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
principalDf['Class'] = y

# Plotting
fig, ax = plt.subplots()
colors = {0: 'blue', 1: 'red'}  # Color code: 0 for 'not fraudulent', 1 for 'fraudulent'

ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'],
           c=principalDf['Class'].apply(lambda x: colors[x]),
           alpha=0.5)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Credit Card Transactions')

# Creating a legend for the colors
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Fraudulent')
blue_patch = mpatches.Patch(color='blue', label='Not Fraudulent')
plt.legend(handles=[red_patch, blue_patch])

plt.show()
