from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
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

tsne = TSNE(n_components=2, random_state=0)
tsne_results = tsne.fit_transform(x)

# Creating a DataFrame with the t-SNE components and the Class
df_tsne = pd.DataFrame(data=tsne_results, columns=['TSNE1', 'TSNE2'])
df_tsne['Class'] = y

# Plotting
colors = {0: 'blue', 1: 'red'}  # Color code: 0 for 'not fraudulent', 1 for 'fraudulent'

plt.figure(figsize=(8,6))
plt.scatter(df_tsne['TSNE1'], df_tsne['TSNE2'],
            c=df_tsne['Class'].apply(lambda x: colors[x]),
            alpha=0.5)

plt.xlabel('TSNE1')
plt.ylabel('TSNE2')
plt.title('t-SNE visualization of the dataset')

# Creating a legend for the colors
red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Fraudulent',
                          markerfacecolor='red', markersize=10)
blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Not Fraudulent',
                        markerfacecolor='blue', markersize=10)
plt.legend(handles=[red_patch, blue_patch])

plt.show()
