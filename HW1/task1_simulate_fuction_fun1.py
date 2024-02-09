import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# simulate a non-linear function
x = np.random.seed(42)

# Generate random data
X_Rand = np.random.rand(300, 1).astype(np.float32)
Y_Rand = np.sin(5 * np.pi * X_Rand)/(5 * np.pi * X_Rand).astype(np.float32)

# Convert data to PyTorch tensors
x = torch.tensor(X_Rand)
y = torch.tensor(Y_Rand)

plt.scatter(x, y, label='Data Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Plot of X and Y')
plt.legend()
plt.show()

# Define a simple DNN model with the same number of parameters for both models
class DNN_Model1(nn.Module):
    def __init__(self):
        super(DNN_Model1, self).__init__()
        self.hidden1 = nn.Linear(1, 5)   
        self.hidden2 = nn.Linear(5, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.hidden4 = nn.Linear(10, 10)
        self.hidden5 = nn.Linear(10, 10)
        self.hidden6 = nn.Linear(10, 10)
        self.hidden7 = nn.Linear(10, 5)
        self.hidden8 = nn.Linear(5, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = torch.relu(self.hidden5(x))
        x = torch.relu(self.hidden6(x))
        x = torch.relu(self.hidden7(x))
        x = self.hidden8(x)
        return x

class DNN_Model2(nn.Module):
    def __init__(self):
        super(DNN_Model2, self).__init__()
        self.hidden1 = nn.Linear(1, 10)  
        self.hidden2 = nn.Linear(10, 18)
        self.hidden3 = nn.Linear(18, 15)
        self.hidden4 = nn.Linear(15, 4)
        self.hidden5 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = torch.relu(self.hidden4(x))
        x = self.hidden5(x)
        return x
    
class DNN_Model3(nn.Module):
    def __init__(self):
        super(DNN_Model3, self).__init__()
        self.hidden1 = nn.Linear(1, 190)   
        self.hidden2 = nn.Linear(190, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x

DNN_Model1 = DNN_Model1()
DNN_Model2 = DNN_Model2()
DNN_Model3 = DNN_Model3()

def train_model(model, X, y, num_epochs=20000, learning_rate=1e-2):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    losses = []
    for epoch in range(num_epochs):
        outputs = model(X)
        loss = criterion(outputs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return losses

print(0)
# Train the models
losses_model1 = train_model(DNN_Model1, x, y)
print(1)
losses_model2 = train_model(DNN_Model2, x, y)
print(2)
losses_model3 = train_model(DNN_Model3, x, y)
print(3)


# Plot the training loss for both models
plt.plot(losses_model1, label='Model 1')
plt.plot(losses_model2, label='Model 2')
plt.plot(losses_model3, label='Model 3')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss of 3 DNN Models')
plt.legend()
plt.show()

print(4)

# Visualize ground-truth and predictions for both models
with torch.no_grad():
    x_values = torch.linspace(0, 1, 100).view(-1, 1)
    predictions_model1 = DNN_Model1(x_values).numpy()
    predictions_model2 = DNN_Model2(x_values).numpy()
    predictions_model3 = DNN_Model3(x_values).numpy()

plt.scatter(x, y, label='Ground Truth')
plt.plot(x_values.numpy(), predictions_model1, label='Model 1 Predictions', linestyle='dashed')
plt.plot(x_values.numpy(), predictions_model2, label='Model 2 Predictions', linestyle='dashed')
plt.plot(x_values.numpy(), predictions_model3, label='Model 3 Predictions', linestyle='dashed')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Ground Truth and Predictions of 3 DNN Models')
plt.legend()
plt.show()