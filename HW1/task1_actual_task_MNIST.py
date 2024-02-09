import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define CNN model
class CNN_Model1(nn.Module):
    def __init__(self):
        super(CNN_Model1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.fc = nn.Linear(16*26*26, 10) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 16*26*26)
        x = self.fc(x)
        return x
    
class CNN_Model2(nn.Module):
    def __init__(self):
        super(CNN_Model2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN_Model3(nn.Module):
    def __init__(self):
        super(CNN_Model3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128*3*3, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 128*3*3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the CNN models, loss function, and optimizer
model1 = CNN_Model1()
model2 = CNN_Model2()
model3 = CNN_Model3()
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=1e-2)
optimizer2 = optim.Adam(model2.parameters(), lr=1e-2)
optimizer3 = optim.Adam(model3.parameters(), lr=1e-2)

# Training loop
num_epochs = 50

# Initialize lists to store training losses and accuracies for each model
losses_model1 = []
accuracies_model1 = []
losses_model2 = []
accuracies_model2 = []
losses_model3 = []
accuracies_model3 = []

def train_model(model, optimizer, criterion, num_epochs, losses, accuracies):
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        accuracy = correct_predictions / total_samples

        # Record training loss and accuracy
        losses.append(epoch_loss)
        accuracies.append(accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

# Train model1
train_model(model1, optimizer1, criterion, num_epochs, losses_model1, accuracies_model1) 

# Train model2
train_model(model2, optimizer2, criterion, num_epochs, losses_model2, accuracies_model2)

# Train model3
train_model(model3, optimizer3, criterion, num_epochs, losses_model3, accuracies_model3)

# Plot Training Loss for 3 models
plt.plot(losses_model1, label='Model1 Loss')
plt.plot(losses_model2, label='Model2 Loss')
plt.plot(losses_model3, label='Model3 Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot Training Accuracy for 3 models
plt.plot(accuracies_model1, label='Model1 Accuracy')
plt.plot(accuracies_model2, label='Model2 Accuracy')
plt.plot(accuracies_model3, label='Model3 Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()