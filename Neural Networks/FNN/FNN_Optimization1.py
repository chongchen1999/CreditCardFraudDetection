import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Load your dataset
# Assuming the DataFrame is named df
df = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\creditcard_2013_clean.csv')

# Separate features and target
X = df.iloc[:, 1:-1].values  # Exclude 'id' and 'Class' columns
y = df['Class'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train[:, None], dtype=torch.float32)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_torch = torch.tensor(y_test[:, None], dtype=torch.float32)

# Calculate class weights
class_weights = torch.tensor([1.0 / y_train.mean(), 1.0 / (1 - y_train.mean())], dtype=torch.float32)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])

# Create TensorDatasets
train_dataset = TensorDataset(X_train_torch, y_train_torch)

# Since we're using class weights in the loss function, we might not need a sampler for balancing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Define the neural network architecture
class CreditCardFraudNN(nn.Module):
    def __init__(self):
        super(CreditCardFraudNN, self).__init__()
        self.layer1 = nn.Linear(29, 64)  # Assuming 29 features as in your dataset
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.output_layer(x)  # Removed the sigmoid activation to use with BCEWithLogitsLoss
        return x


model = CreditCardFraudNN()

# Specify the optimizer
optimizer = optim.Adam(model.parameters())

# Training loop
num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}')

# Evaluation with a modified decision threshold
model.eval()
threshold = 0.5  # Adjust this threshold based on your precision/recall trade-off needs
with torch.no_grad():
    y_pred_probs = torch.sigmoid(model(X_test_torch))  # Apply sigmoid to output since we used BCEWithLogitsLoss
    y_pred = (y_pred_probs > threshold).float()
    print(classification_report(y_test, y_pred.numpy(), digits=4, zero_division=0))
