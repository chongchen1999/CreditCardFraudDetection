import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r'/dataset/creditcard_2013_clean.csv')

# Separate features and target
X = df.iloc[:, 1:-1]  # Exclude 'id' and 'Class' columns
y = df['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_rnn = Sequential([
    SimpleRNN(64, input_shape=(X_train_scaled.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    SimpleRNN(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model_rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape data for RNN layer
X_train_rnn = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_rnn = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

model_rnn.fit(X_train_rnn, y_train, epochs=50, batch_size=32, validation_split=0.2)

y_pred_probs = model_rnn.predict(X_test_scaled)
SNN_Prediction = np.round(y_pred_probs).astype(int)  # Convert probabilities to binary predictions

print(classification_report(y_test, SNN_Prediction, digits=4))