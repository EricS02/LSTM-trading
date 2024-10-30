import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from bayes_opt import BayesianOptimization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\E\LSTM\Stock_data\enhanced_combined_data_with_labels.csv')

features = ['Volume', 'High', 'Low', 'Close', 'Price_Change','Gain','Loss','Avg_Gain','Avg_Loss','Bullish','Neutral','Bearish','Sentiment_Diff']

sequence_length = 10
data_features = data[features].values
data_labels = data['Decision6'].values

# Convert to sequences
data_features_sequences = []
data_labels_sequences = []

for i in range(len(data_features) - sequence_length):
    data_features_sequences.append(data_features[i:i+sequence_length])
    data_labels_sequences.append(data_labels[i+sequence_length])

# Convert lists to numpy arrays
data_features_sequences = np.array(data_features_sequences)
data_labels_sequences = np.array(data_labels_sequences)

# Determine the number of sequences
total_sequences = len(data_features_sequences)

# Define split sizes
train_size = int(0.7 * total_sequences)
val_size = int(0.15 * total_sequences)
# The remaining data will be used for testing


# Split data
train_sequences, train_labels = data_features_sequences[:train_size], data_labels_sequences[:train_size]
val_sequences, val_labels = data_features_sequences[train_size:train_size+val_size], data_labels_sequences[train_size:train_size+val_size]
test_sequences, test_labels = data_features_sequences[train_size+val_size:], data_labels_sequences[train_size+val_size:]

print(train_sequences.shape)
print(train_labels.shape)


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_()
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 
        return out

input_dim = 13  # Number of input features
hidden_dim = 32
layer_dim = 1   # Only one LSTM layer
output_dim = 3  # 'buy', 'sell', 'hold'

model = LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

SEQUENCE_LENGTH = 10

data_features_sequences = []
data_labels_sequences = []

for i in range(len(data_features) - SEQUENCE_LENGTH):
    data_features_sequences.append(data_features[i:i+SEQUENCE_LENGTH])
    data_labels_sequences.append(data_labels[i+SEQUENCE_LENGTH])

data_features_sequences = np.array(data_features_sequences)
data_labels_sequences = np.array(data_labels_sequences)


# Split the sequences into training, validation, and test sets
train_size = int(0.7 * len(data_features_sequences))
val_size = int(0.2 * len(data_features_sequences))
test_size = len(data_features_sequences) - train_size - val_size

train_features, val_features, test_features = data_features_sequences[:train_size], data_features_sequences[train_size:train_size+val_size], data_features_sequences[train_size+val_size:]
train_labels, val_labels, test_labels = data_labels_sequences[:train_size], data_labels_sequences[train_size:train_size+val_size], data_labels_sequences[train_size+val_size:]


# Convert data to PyTorch tensors
train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_labels, dtype=torch.float32)
val_features_tensor = torch.tensor(val_features, dtype=torch.float32)
val_labels_tensor = torch.tensor(val_labels, dtype=torch.float32)

# Create DataLoader objects
train_dataset = TensorDataset(train_features_tensor, train_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Training function
def train(model, optimizer, data_loader, loss_fn):
    model.train()
    total_loss = 0

    for seq_data, seq_labels in data_loader:
        optimizer.zero_grad()
        outputs = model(seq_data)
        loss = loss_fn(outputs, seq_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss

# Validation function
def validate(model, data_loader, loss_fn):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for seq_data, seq_labels in data_loader:
            outputs = model(seq_data)
            loss = loss_fn(outputs, seq_labels)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    return avg_loss



# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    true_labels = []
    predictions = []

    with torch.no_grad():
        for seq_data, seq_labels in data_loader:
            outputs = model(seq_data)
            _, predicted = torch.max(outputs.data, 1)
            true_labels.extend(seq_labels.numpy())
            predictions.extend(predicted.numpy())

    return accuracy_score(true_labels, predictions)

# Assuming the sequences and corresponding labels are set up correctly:
train_data = TensorDataset(train_sequences, train_labels)
val_data = TensorDataset(val_sequences, val_labels)
test_data = TensorDataset(test_sequences, test_labels)

# Create DataLoaders
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Training loop
num_epochs = 100
training_loss_values = []
validation_loss_values = []
training_accuracy_values = []
validation_accuracy_values = []

for epoch in range(num_epochs):
    LSTMModel.train()
    train_loss = 0.0
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = LSTMModel(batch_data)
        loss = loss_fn(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    train_loss = train_loss / len(train_loader)
    training_loss_values.append(train_loss)
    
    LSTMModel.eval()
    val_loss = 0.0
    for batch_data, batch_labels in val_loader:
        outputs = LSTMModel(batch_data)
        loss = loss_fn(outputs, batch_labels)
        val_loss += loss.item()
        
    val_loss = val_loss / len(val_loader)
    validation_loss_values.append(val_loss)

    train_accuracy = calculate_accuracy(LSTMModel, train_loader)
    val_accuracy = calculate_accuracy(LSTMModel, val_loader)
    training_accuracy_values.append(train_accuracy)
    validation_accuracy_values.append(val_accuracy)

    print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Accuracy: {val_accuracy:.2f}%')

# Evaluate the model's accuracy on the validation set
validation_accuracy = calculate_accuracy(LSTMModel, val_loader)
print(f'Model Accuracy on Validation Set: {validation_accuracy:.2f}%')

# Evaluate the model's accuracy on the test set
test_accuracy = calculate_accuracy(LSTMModel, test_loader)
print(f'Model Accuracy on Test Set: {test_accuracy:.2f}%')

# Plotting the loss
plt.figure()
plt.plot(training_loss_values, label='Training Loss')
plt.plot(validation_loss_values, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plotting the accuracy
plt.figure()
plt.plot(training_accuracy_values, label='Training Accuracy')
plt.plot(validation_accuracy_values, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


torch.save(LSTMModel.state_dict(), (r'C:\Users\E\LSTM\Weights'))
print("Model training completed and trained model saved.")