import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader_64 = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader_64 = DataLoader(test_dataset, batch_size=64, shuffle=False)
train_loader_1024 = DataLoader(train_dataset, batch_size=1024, shuffle=True)
test_loader_1024 = DataLoader(test_dataset, batch_size=1024, shuffle=False)

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

# Function to train a model
def train_model(model, train_loader, test_loader, optimizer, criterion, epochs):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
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
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Evaluate on test set
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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    return train_losses, test_losses, train_accuracies, test_accuracies

# Function to interpolate between two models
def interpolate_models(m1, m2, alpha):
    interpolated_model = DNN_Model()
    m1_params = m1.state_dict()
    m2_params = m2.state_dict()
    interp_params = {}
    for key in m1_params.keys():
        interp_params[key] = (1 - alpha) * m1_params[key] + alpha * m2_params[key]
    interpolated_model.load_state_dict(interp_params)
    return interpolated_model

# Train models with different parameters
m1 = DNN_Model().to(device)
m2 = DNN_Model().to(device)

criterion = nn.CrossEntropyLoss()

# Interpolation parameters
alphas = [i / 10 for i in range(-10, 21)]

# Train interpolated models and record losses
interpolated_losses_train_64_e2 = []
interpolated_losses_train_1024_e2 = []
interpolated_losses_test_64_e2 = []
interpolated_losses_test_1024_e2 = []
interpolated_losses_train_64_e3 = []
interpolated_losses_test_64_e3 = []

interpolated_acc_train_64_e2 = []
interpolated_acc_train_1024_e2 = []
interpolated_acc_test_64_e2 = []
interpolated_acc_test_1024_e2 = []
interpolated_acc_train_64_e3 = []
interpolated_acc_test_64_e3 = []

for alpha in alphas:
    print(f"Alpha = {alpha}")
    interpolated_model = interpolate_models(m1, m2, alpha)
    optimizer_e2 = optim.Adam(interpolated_model.parameters(), lr=1e-2)
    optimizer_e3 = optim.Adam(interpolated_model.parameters(), lr=1e-2)
    train_loss_64_e2, test_loss_64_e2, train_acc_64_e2, test_acc_64_e2 = train_model(interpolated_model, train_loader_64, test_loader_64,optimizer_e2, criterion, 10)
    train_loss_1024_e2, test_loss_1024_e2, train_acc_1024_e2, test_acc_1024_e2 = train_model(interpolated_model, train_loader_1024, test_loader_1024, optimizer_e2, criterion, 10)
    train_loss_64_e3, test_loss_64_e3, train_acc_64_e3, test_acc_64_e3 = train_model(interpolated_model, train_loader_64, test_loader_64,optimizer_e3, criterion, 10)

    interpolated_losses_train_64_e2.append(train_loss_64_e2[-1])
    interpolated_losses_train_1024_e2.append(train_loss_1024_e2[-1])
    interpolated_losses_train_64_e3.append(train_loss_64_e3[-1])
    interpolated_losses_test_64_e2.append(test_loss_64_e2[-1])
    interpolated_losses_test_1024_e2.append(test_loss_1024_e2[-1])
    interpolated_losses_test_64_e3.append(test_loss_64_e3[-1])

    interpolated_acc_train_64_e2.append(train_acc_64_e2[-1])
    interpolated_acc_train_1024_e2.append(train_acc_1024_e2[-1])
    interpolated_acc_train_64_e3.append(train_acc_64_e3[-1])
    interpolated_acc_test_64_e2.append(test_acc_64_e2[-1])
    interpolated_acc_test_1024_e2.append(test_acc_1024_e2[-1])
    interpolated_acc_test_64_e3.append(test_acc_64_e3[-1])

# Plotting
fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Interpolation Ratio')
ax1.set_ylabel('Cross-Entropy Loss', color=color)
ax1.plot(alphas, interpolated_losses_train_64_e2, linestyle='-', label='Train_64_e2_loss', color='blue')
ax1.plot(alphas, interpolated_losses_test_64_e2, linestyle='--', label='Test_64_e2_loss', color='blue')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Accuracy', color=color)  
ax2.plot(alphas, interpolated_acc_train_64_e2, linestyle='-', label='Train_64_e2_accuracy', color='red')
ax2.plot(alphas, interpolated_acc_test_64_e2, linestyle='--', label='Test_64_e2_accuracy', color='red')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Loss and Accuracy vs Interpolation Ratio')
plt.show()

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Interpolation Ratio')
ax1.set_ylabel('Cross-Entropy Loss', color=color)
ax1.plot(alphas, interpolated_losses_train_1024_e2, linestyle='-', label='Train_1024_e2_loss', color='blue')
ax1.plot(alphas, interpolated_losses_test_1024_e2, linestyle='--', label='Test_1024_e2_loss', color='blue')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Accuracy', color=color)  
ax2.plot(alphas, interpolated_acc_train_1024_e2, linestyle='-', label='Train_1024_e2_accuracy', color='red')
ax2.plot(alphas, interpolated_acc_test_1024_e2, linestyle='--', label='Test_1024_e2_accuracy', color='red')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Loss and Accuracy vs Interpolation Ratio')
plt.show()

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Interpolation Ratio')
ax1.set_ylabel('Cross-Entropy Loss', color=color)
ax1.plot(alphas, interpolated_losses_train_64_e3, linestyle='-', label='Train_64_e3_loss', color='blue')
ax1.plot(alphas, interpolated_losses_test_64_e3, linestyle='--', label='Test_64_e3_loss', color='blue')
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Accuracy', color=color)  
ax2.plot(alphas, interpolated_acc_train_64_e3, linestyle='-', label='Train_64_e3_accuracy', color='red')
ax2.plot(alphas, interpolated_acc_test_64_e3, linestyle='--', label='Test_64_e3_accuracy', color='red')
ax2.tick_params(axis='y', labelcolor=color)
ax2.legend(loc='upper right')

fig.tight_layout()  
plt.title('Loss and Accuracy vs Interpolation Ratio')
plt.show()