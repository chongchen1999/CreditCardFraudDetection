import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# Load your dataset
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

X_train = df_train.iloc[:, 1: -1].values
y_train = df_train['Class'].values
X_test = df_test.iloc[:, 1: -1].values
y_test = df_test['Class'].values

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure all tensor data is sent to the appropriate device
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_torch = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_torch = torch.tensor(y_test[:, None], dtype=torch.float32).to(device)

# TensorDatasets and DataLoader initialization remains the same
train_dataset = TensorDataset(X_train_torch, y_train_torch)
test_dataset = TensorDataset(X_test_torch, y_test_torch)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Neural network architecture with a modification for CUDA
class CreditCardFraudNN(nn.Module):
    def __init__(self):
        super(CreditCardFraudNN, self).__init__()
        self.layer1 = nn.Linear(29, 64)
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(64, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.output_layer(x))
        return x

# Initialize the model and move it to CUDA
model = CreditCardFraudNN().to(device)

# Compile the model with the same loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model with modifications for CUDA
num_epochs = 75
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Evaluate the model, ensuring to move the model to eval mode and use no_grad context
model.eval()
with torch.no_grad():
    y_pred_probs = model(X_test_torch)
    y_pred = torch.round(y_pred_probs).cpu().numpy()  # Move predictions back to CPU for sklearn metrics
    print(classification_report(y_test, y_pred, digits=4))
