from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This is needed for 3D plotting
import pandas as pd

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

# Applying PCA to reduce to 3 components
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(x)

# Creating a DataFrame with the principal components and the Class
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2', 'principal component 3'])
principalDf['Class'] = y

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # This creates a 3D plot

colors = {0: 'blue', 1: 'red'}  # Color code: 0 for 'not fraudulent', 1 for 'fraudulent'

ax.scatter(principalDf['principal component 1'], principalDf['principal component 2'], principalDf['principal component 3'],
           c=principalDf['Class'].apply(lambda x: colors[x]),
           alpha=0.5)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.title('3D PCA of Credit Card Transactions')

# Creating a legend for the colors
red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Fraudulent',
                          markerfacecolor='red', markersize=10)
blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Not Fraudulent',
                        markerfacecolor='blue', markersize=10)
plt.legend(handles=[red_patch, blue_patch])

plt.show()
