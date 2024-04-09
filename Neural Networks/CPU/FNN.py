import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Assuming X_train, y_train, X_test, y_test are already defined
# Load your dataset
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'/dataset/creditcard_2013_clean.csv')

# Separate features and target
X = df.iloc[:, 1:-1]  # Exclude 'id' and 'Class' columns
y = df['Class']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Define the model
model = Sequential([
    Dense(64, input_dim=29, activation='relu'),  # 28 anonymized features + 1 'Amount' feature
    Dropout(0.5),  # Dropout layer to prevent overfitting (adjust the rate as needed)
    Dense(128, activation='relu'),
    Dropout(0.5),  # Another dropout layer for added regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),  # You can adjust the learning rate
              loss='binary_crossentropy',  # Suitable for binary classification
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Fit the model to the training data
history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=50,  # You might need to adjust the number of epochs
                    batch_size=32,  # And the batch size
                    verbose=2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_accuracy}, Test loss: {test_loss}')
