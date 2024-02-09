import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

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
class DNN_Model(nn.Module):
    def __init__(self):
        super(DNN_Model, self).__init__()
        self.hidden1 = nn.Linear(1, 128)
        self.hidden2 = nn.Linear(128, 1)   

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x

model = DNN_Model()

def train_model(model, x, y, num_epochs, optimizer):
    criterion = nn.MSELoss()

    all_epoch = []
    all_loss = []
    all_gradients_norm = []

    converged = False

    for epoch in range(num_epochs):
        if(converged):
            return all_epoch, all_loss, prediction, all_gradients_norm

        prediction = model(x)
        loss = criterion(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        all_epoch.append(epoch)
        all_loss.append(loss.detach().numpy())

        #Calculate the gradiet
        grad_all = 0.0

        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5

        all_gradients_norm.append(grad_norm)

        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

        if (epoch > 5) and  (all_loss[-1] < 0.001):
            if abs(all_loss[-3] - all_loss[-2]) < 1.0e-05 and abs(all_loss[-2] - all_loss[-1]) < 1.0e-05:
                print("Convergence: ",all_loss[-1])
                # converged = True

    return all_epoch, all_loss, prediction, all_gradients_norm

optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
all_epoch, all_loss, prediction, all_gradients_norm = train_model(model, x, y, 2000, optimizer)

# Epoch & loss
plt.plot(all_epoch, all_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Epoch & gradient
plt.plot(all_epoch, all_gradients_norm)
plt.xlabel('Epoch')
plt.ylabel('Gradient')
plt.show()