import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load datasets
df_train = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\train_dataset_1to100.csv')
df_test = pd.read_csv(r'E:\01 - Northeastern\2024 Spring\01 - Data Science\Final Project\dataset\test_dataset_1to100.csv')

# Splitting the datasets into features and target variable
X_train = df_train.iloc[:, 1: -1]
y_train = df_train['Class']
X_test = df_test.iloc[:, 1: -1]
y_test = df_test['Class']

from imblearn.over_sampling import SVMSMOTE

# Create an instance of SVMSMOTE
svmsmote = SVMSMOTE(sampling_strategy='auto', random_state=42)

# Fit the model and apply the oversampling
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)

# Check the new class distribution
print('Before SVMSMOTE:', y_train.value_counts())
print('After SVMSMOTE:', y_train_resampled.value_counts())


# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(1)  # Add channel dimension
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.float32).unsqueeze(1)  # Add dimension for compatibility

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3)  # Reduced number of filters, smaller kernel
        self.pool = nn.MaxPool1d(2)  # Pooling to reduce size by half
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)  # Additional Conv layer for more complex features
        # Calculate the size here dynamically based on input size, or use a fixed value based on calculation
        self.flatten = nn.Flatten()
        # Assuming the original size is 29, after first conv and pool: ((29-3+1)/2)
        # After second conv with same padding: (((29-3+1)/2)-3+1)
        # Adjust the input features of fc1 accordingly, considering the number of output channels from the last conv layer
        self.fc1 = nn.Linear(128 * (((29-3+1)//2)-3+1), 64)  # Adjusted input features
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Initialize model, loss, and optimizer
model = CNNModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
model.train()
for epoch in range(75):  # Loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch + 1}, loss: {running_loss / len(train_loader)}')

print('Finished Training')

# Evaluate the model
model.eval()
y_pred_list = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predicted = torch.round(outputs.cpu().data)
        y_pred_list.append(predicted)

y_pred = torch.cat(y_pred_list).numpy()
print(classification_report(y_test, y_pred, digits=4))

C_a = 50.0

y_test_flat = y_test.values
CNN_Prediction_flat = y_pred.flatten()
Amounts = df_test['Amount'].values  # Make sure this is a 1D array with the same length as y_test

Cost = np.sum(y_test_flat * (1 - CNN_Prediction_flat) * Amounts + C_a * CNN_Prediction_flat)
Cost_t = np.sum(y_test_flat * Amounts)

Savings = 1 - (Cost / Cost_t)
print(f"Savings: {Savings}")