import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader_64 = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader_256 = DataLoader(train_dataset, batch_size=256, shuffle=True)
train_loader_1024 = DataLoader(train_dataset, batch_size=1024, shuffle=True)
train_loader_4096 = DataLoader(train_dataset, batch_size=4096, shuffle=True)
train_loader_8192 = DataLoader(train_dataset, batch_size=8192, shuffle=True)

test_loader_64 = DataLoader(test_dataset, batch_size=64, shuffle=False)
test_loader_256 = DataLoader(test_dataset, batch_size=256, shuffle=False)
test_loader_1024 = DataLoader(test_dataset, batch_size=1024, shuffle=False)
test_loader_4096 = DataLoader(test_dataset, batch_size=4096, shuffle=False)
test_loader_8192 = DataLoader(test_dataset, batch_size=8192, shuffle=False)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define DNN model
class DNN_Model(nn.Module):
    def __init__(self):
        super(DNN_Model, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_test_model(model, train_loader, test_loader, optimizer, criterion, epochs, device):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    sensitivity = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        froGrad = 0.0
        count = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            _, predicted_train = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted_train == labels).sum().item()

            for p in model.parameters():
                if p.grad is not None:
                    grad = p.grad
                    froGrad_norm = torch.linalg.norm(grad).item()
                    froGrad += froGrad_norm
                    count += 1

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        if count > 0:
            sensitivity.append(froGrad / count)
        else:
            sensitivity.append(0.0)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Sensitivity: {sensitivity[-1]:.4f}")

        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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

        print(f"Epoch {epoch+1}/{epochs}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, train_accuracies, test_losses, test_accuracies, sensitivity

# Define criterion and optimizer
criterion = nn.CrossEntropyLoss()

# Train models with different training approaches
train_losses_list = []
test_losses_list = []
train_accuracies_list = []
test_accuracies_list = []
sensitivities_list = []

model = DNN_Model().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# Model 1: Batch size 64
train_losses1, train_accuracies1, test_losses1, test_accuracies1, sensitivities1 = train_test_model(model, train_loader_64, test_loader_64, optimizer, criterion, epochs=30, device=device)
train_losses_list.append(train_losses1)
test_losses_list.append(test_losses1)
train_accuracies_list.append(train_accuracies1)
test_accuracies_list.append(test_accuracies1)
sensitivities_list.append(sensitivities1)

# Model 2: Batch size 256
train_losses2, train_accuracies2, test_losses2, test_accuracies2, sensitivities2 = train_test_model(model, train_loader_256, test_loader_256, optimizer, criterion, epochs=30, device=device)
train_losses_list.append(train_losses2)
test_losses_list.append(test_losses2)
train_accuracies_list.append(train_accuracies2)
test_accuracies_list.append(test_accuracies2)
sensitivities_list.append(sensitivities2)

# Model 3: Batch size 1024
train_losses3, train_accuracies3, test_losses3, test_accuracies3, sensitivities3 = train_test_model(model, train_loader_1024, test_loader_1024, optimizer, criterion, epochs=30, device=device)
train_losses_list.append(train_losses3)
test_losses_list.append(test_losses3)
train_accuracies_list.append(train_accuracies3)
test_accuracies_list.append(test_accuracies3)
sensitivities_list.append(sensitivities3)

# Model 4: Batch size 4096
train_losses4, train_accuracies4, test_losses4, test_accuracies4, sensitivities4 = train_test_model(model, train_loader_4096, test_loader_4096, optimizer, criterion, epochs=30, device=device)
train_losses_list.append(train_losses4)
test_losses_list.append(test_losses4)
train_accuracies_list.append(train_accuracies4)
test_accuracies_list.append(test_accuracies4)
sensitivities_list.append(sensitivities4)

# Model 5: Batch size 8192
train_losses5, train_accuracies5, test_losses5, test_accuracies5, sensitivities5 = train_test_model(model, train_loader_8192, test_loader_8192, optimizer, criterion, epochs=30, device=device)
train_losses_list.append(train_losses5)
test_losses_list.append(test_losses5)
train_accuracies_list.append(train_accuracies5)
test_accuracies_list.append(test_accuracies5)
sensitivities_list.append(sensitivities5)

# Plotting
fig, ax1 = plt.subplots()
color = 'blue'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Loss', color='blue')
ax1.plot([64, 256, 1024, 4096, 8192], [train_losses_list[i][-1] for i in range(5)], linestyle='-', label='Train Loss', color=color)
ax1.plot([64, 256, 1024, 4096, 8192], [test_losses_list[i][-1] for i in range(5)], linestyle='--', label='Test Loss', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Sensitivity', color=color)  
ax2.plot([64, 256, 1024, 4096, 8192], [sensitivities_list[i][-1] for i in range(5)], linestyle='-', label='Sensitivity', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Loss and Sensitivity vs Batch Size')
plt.show()

fig, ax1 = plt.subplots()
color = 'blue'
ax1.set_xlabel('Batch Size')
ax1.set_ylabel('Accuracy', color=color)
ax1.plot([64, 256, 1024, 4096, 8192], [train_accuracies_list[i][-1] for i in range(5)], linestyle='-', label='Train Accuracy', color=color)
ax1.plot([64, 256, 1024, 4096, 8192], [test_accuracies_list[i][-1] for i in range(5)], linestyle='--', label='Test Accuracy', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'red'
ax2.set_ylabel('Sensitivity', color=color)  
ax2.plot([64, 256, 1024, 4096, 8192], [sensitivities_list[i][-1] for i in range(5)], linestyle='-', label='Sensitivity', color=color)
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Accuracy and Sensitivity vs Batch Size')
plt.show()