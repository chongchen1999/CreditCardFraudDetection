from cuml.manifold import TSNE
import cudf
import pandas as pd

df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\creditcard_2023.csv')

# Assuming `df` is your DataFrame with the V1-V28 features and 'Class' as the target variable
features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
            'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28']

# Convert your dataframe to a cuDF DataFrame for GPU acceleration
gdf = cudf.DataFrame(df)

# Separating out the features and target
x_gpu = gdf.loc[:, features].values
y_gpu = gdf.loc[:, 'Class'].values

# No need to standardize features separately, TSNE will handle it

# Configuring t-SNE for 3 components (3D), on GPU
tsne_gpu = TSNE(n_components=3, method='barnes_hut', random_state=0)
tsne_results_gpu = tsne_gpu.fit_transform(x_gpu)

# Creating a DataFrame with the t-SNE components and the Class
df_tsne_gpu = pd.DataFrame(data=tsne_results_gpu.to_array(), columns=['TSNE1', 'TSNE2', 'TSNE3'])
df_tsne_gpu['Class'] = y_gpu.to_array()

# You can then proceed to plot this data as before
