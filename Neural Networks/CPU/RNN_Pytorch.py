import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load dataset (ensure to change the path as necessary)
df = pd.read_csv(r'/dataset/creditcard_2013_clean.csv')

# Separate features and target
X = df.iloc[:, 1:-1].values  # Assuming the first column is 'id' and last column is 'Class'
y = df['Class'].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)  # Add channel dimension
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# Define the RNN model in PyTorch
class RNNModel(nn.Module):
    def __init__(self, input_dim):
        super(RNNModel, self).__init__()
        self.rnn1 = nn.RNN(input_dim, 64, batch_first=True, nonlinearity='relu')
        self.dropout1 = nn.Dropout(0.2)
        self.rnn2 = nn.RNN(64, 64, batch_first=True, nonlinearity='relu')
        self.dropout2 = nn.Dropout(0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.rnn1(x)
        x = self.dropout1(x)
        x, _ = self.rnn2(x)
        x = self.dropout2(x[:, -1, :])  # Only take the output from the last RNN cell
        x = self.fc(x)
        return torch.sigmoid(x)


# Instantiate the model, loss function, and optimizer
model = RNNModel(input_dim=1)
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert datasets to DataLoader for batch processing
train_dataset = TensorDataset(X_train_scaled, y_train)
test_dataset = TensorDataset(X_test_scaled, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Training the model
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
all_preds = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        all_preds.extend(outputs.squeeze().round().int().numpy())

# Print classification report
print(classification_report(y_test.numpy(), np.array(all_preds), digits=4))
