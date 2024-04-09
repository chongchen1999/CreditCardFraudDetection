import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import plot_model

# Assuming X_train_scaled is defined elsewhere and is your input data
# Example input shape of 10 features for the input layer
input_shape = (10,)

model = Sequential([
    Dense(16, activation='relu', input_shape=input_shape),  # Input layer with shape (10,)
    Dropout(0.2),  # Dropout layer for regularization
    Dense(16, activation='relu'),  # Hidden layer
    Dense(1, activation='sigmoid')  # Output layer
])

# Now, visualize the model. This will save an image named 'model_visualization.png' in your current working directory.
plot_model(model, to_file='model_visualization.png', show_shapes=True, show_layer_names=True)
