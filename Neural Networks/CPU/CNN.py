import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load your dataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Separate features and target
X_train = df.iloc[:, 1:-1]  # Exclude 'id' and 'Class' columns
y_train = df['Class']

X_test = df_test.iloc[:, 1:-1]
y_test = df_test['Class'].astype(int).values.reshape(-1, 1)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_cnn = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for Conv1D layer
X_train_cnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_cnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model_cnn.fit(X_train_cnn, y_train, epochs=75, batch_size=128, validation_split=0.2)

y_pred_probs = model_cnn.predict(X_test_scaled)
CNN_Prediction = np.round(y_pred_probs).astype(int)  # Convert probabilities to binary predictions

print(classification_report(y_test, CNN_Prediction, digits=4))

C_a = 50.0

# Convert all arrays to 1D arrays to ensure element-wise operations
y_test_flat = y_test.flatten()
CNN_Prediction_flat = CNN_Prediction.flatten()
Amounts = df_test['Amount'].values  # Make sure this is a 1D array with the same length as y_test

# Calculate Cost
# The element-wise multiplication (*) should not result in an array larger than the number of samples
Cost = np.sum(y_test_flat * (1 - CNN_Prediction_flat) * Amounts + C_a * CNN_Prediction_flat)

# Calculate Cost_t, which is the cost of classifying everything as not fraud
Cost_t = np.sum(y_test_flat * Amounts)

# Calculate Savings
Savings = 1 - (Cost / Cost_t)

print(f"Savings: {Savings}")