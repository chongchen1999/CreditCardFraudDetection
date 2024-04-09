import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# Dataset Import
df_train = pd.read_csv(
    r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(
    r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

X_train = df_train.iloc[:, 1:-1].values
y_train = df_train['Class'].values
X_test = df_test.iloc[:, 1:-1].values
y_test = df_test['Class'].values

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class FraudDetectionEnv:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.current_index = 0

    def reset(self):
        self.current_index = 0
        return self.X[self.current_index]

    def step(self, action):
        correct_action = self.y[self.current_index]
        reward = 1 if action == correct_action else -1
        self.current_index += 1
        done = self.current_index == len(self.X)
        next_state = self.X[self.current_index] if not done else None
        return next_state, reward, done


# Define Q-learning training function
def q_learning(env, episodes, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
    q_table = np.zeros((env.X.shape[0], 2))  # Simplified Q-table
    for episode in range(episodes):
        state = env.reset()
        done = False
        while not done:
            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1])  # Explore
            else:
                action = np.argmax(q_table[env.current_index])  # Exploit
            next_state, reward, done = env.step(action)

            # Update Q-value
            old_value = q_table[env.current_index, action]
            next_max = np.max(q_table[env.current_index]) if not done else 0
            new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
            q_table[env.current_index, action] = new_value

    return q_table


# Training the model
env = FraudDetectionEnv(X_train_scaled, y_train)
q_table = q_learning(env, 1000)  # Number of episodes

# Model Evaluation
env_test = FraudDetectionEnv(X_test_scaled, y_test)
state = env_test.reset()
preds = []
preds_proba = []

while state is not None:
    action = np.argmax(q_table[env_test.current_index])  # Use trained Q-table for predictions
    preds.append(action)
    # Using the Q-values as proxy probabilities
    q_values = q_table[env_test.current_index]
    proba = np.exp(q_values) / np.sum(np.exp(q_values))  # Softmax to derive probabilities
    preds_proba.append(proba[1])  # Assuming class '1' probability
    state, _, done = env_test.step(action)

print(classification_report(y_test, preds, digits=4))
roc_auc = roc_auc_score(y_test, preds_proba)
print(f'ROC AUC Score: {roc_auc}')
