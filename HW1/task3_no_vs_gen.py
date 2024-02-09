import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define DNN model, use num_parameters to contral the number of parameters
class DNN_Model(nn.Module):
    def __init__(self, num_parameters):
        super(DNN_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, num_parameters)
        self.fc2 = nn.Linear(num_parameters, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def train_and_test(model, train_loader, test_loader, optimizer, criterion, epochs=5):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                
                _, predicted_test = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted_test == labels).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        test_accuracy = correct_test / total_test
        test_accuracies.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies

# Define list of different number of parameters
num_parameters_list = [10, 50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800]

train_losses_list = []
test_losses_list = []
train_accuracies_list = []
test_accuracies_list = []
num_parameters_model_list = []

for num_parameters in num_parameters_list:
    model = DNN_Model(num_parameters)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"Model with {num_parameters} parameters")
    print(f"Total number of parameters: {count_parameters(model)}")

    num_parameters_model_list.append(count_parameters(model))

    train_losses, test_losses, train_accuracies, test_accuracies = train_and_test(model, train_loader, test_loader, optimizer, criterion)
    
    train_losses_list.append(train_losses[-1])  # Append only the final loss
    test_losses_list.append(test_losses[-1])    # Append only the final loss
    train_accuracies_list.append(train_accuracies[-1])  # Append only the final accuracy
    test_accuracies_list.append(test_accuracies[-1])    # Append only the final accuracy

# Plotting Loss vs Parameters
plt.scatter(num_parameters_model_list, train_losses_list, label='Training Loss')
plt.scatter(num_parameters_model_list, test_losses_list, label='Testing Loss')
plt.xlabel('Number of Parameters')
plt.ylabel('Loss')
plt.title('Loss vs Number of Parameters')
plt.legend()
plt.show()

# Plotting Accuracy vs Parameters
plt.scatter(num_parameters_list, train_accuracies_list, label='Training Accuracy')
plt.scatter(num_parameters_list, test_accuracies_list, label='Testing Accuracy')
plt.xlabel('Number of Parameters')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Parameters')
plt.legend()
plt.show()