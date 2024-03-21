import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset
dataset = MNIST(root='data/', download=True, transform=ToTensor())

# Analyze dataset distribution
labels = pd.Series([dataset[i][1] for i in range(len(dataset))])
label_distribution = labels.value_counts().sort_index()
print(label_distribution)

# Split dataset into training and validation sets using sklearn
train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42)
train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)

# Create DataLoader for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = Net().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize a DataFrame to log metrics
metrics_df = pd.DataFrame(columns=['Epoch', 'Training Loss'])

# Training the model
def train_model(num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
        metrics_df.loc[epoch] = [epoch+1, avg_loss]

# Evaluate the model
def evaluate_model(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

# Execute training and evaluation
train_model()
evaluate_model(val_loader)

# Plot training loss
metrics_df.plot(x='Epoch', y='Training Loss', title='Training Loss Over Epochs', legend=False)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.show()

# Plot label distribution
label_distribution.plot(kind='bar', title='Label Distribution in MNIST Dataset')
plt.xlabel('Digit Label')
plt.ylabel('Frequency')
plt.show()
