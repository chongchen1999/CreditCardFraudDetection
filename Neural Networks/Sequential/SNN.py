import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load your dataset
df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\creditcard_2013_clean.csv')

# Separate features and target
X = df.iloc[:, 1:-1]  # Exclude 'id' and 'Class' columns
y = df['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model construction
'''
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
'''

model = Sequential([
    Dense(64, input_dim=29, activation='relu'),  # 28 anonymized features + 1 'Amount' feature
    Dropout(0.5),  # Dropout layer to prevent overfitting (adjust the rate as needed)
    Dense(128, activation='relu'),
    Dropout(0.5),  # Another dropout layer for added regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=75, batch_size=32, validation_split=0.2)

y_pred_probs = model.predict(X_test_scaled)
SNN_Prediction = np.round(y_pred_probs).astype(int)  # Convert probabilities to binary predictions

print(classification_report(y_test, SNN_Prediction, digits=4))
