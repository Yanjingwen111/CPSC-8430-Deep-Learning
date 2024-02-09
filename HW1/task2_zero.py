import torch
from torch.autograd import grad
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np

def get_hessian_eigenvectors(model, loss_fn, train_data_loader, num_batches, device, n_top_vectors, param_extract_fn):
    param_extract_fn = param_extract_fn or (lambda x: x.parameters())
    num_params = sum(p.numel() for p in param_extract_fn(model))
    subset_images, subset_labels = [], []
    for batch_idx, (images, labels) in enumerate(train_data_loader):
        if batch_idx >= num_batches:
            break
        subset_images.append(images.to(device))
        subset_labels.append(labels.to(device))
    subset_images = torch.cat(subset_images)
    subset_labels = torch.cat(subset_labels)

    def compute_loss():
        output = model(subset_images)
        return loss_fn(output, subset_labels)

    def hessian_vector_product(vector):
        model.zero_grad()
        grad_params = grad(compute_loss(), param_extract_fn(model), create_graph=True)
        flat_grad = torch.cat([g.view(-1) for g in grad_params])
        grad_vector_product = torch.sum(flat_grad * vector)
        hvp = grad(grad_vector_product, param_extract_fn(model), retain_graph=True)
        return torch.cat([g.contiguous().view(-1) for g in hvp])

    def matvec(v):
        v_tensor = torch.tensor(v, dtype=torch.float32, device=device)
        return hessian_vector_product(v_tensor).cpu().detach().numpy()

    linear_operator = LinearOperator((num_params, num_params), matvec=matvec)
    eigenvalues, eigenvectors = eigsh(linear_operator, k=n_top_vectors, tol=0.001, which='LM', return_eigenvectors=True)
    eigenvectors = np.transpose(eigenvectors)
    return eigenvalues, eigenvectors

# Simulate a non-linear function
seed = 42
np.random.seed(seed)
# Generate random data
X_Rand = np.random.rand(300, 1).astype(np.float32)
Y_Rand = np.sin(5 * np.pi * X_Rand)/(5 * np.pi * X_Rand).astype(np.float32)

# Convert data to PyTorch tensors
x = torch.tensor(X_Rand)
y = torch.tensor(Y_Rand)

# Define a simple DNN model
class DNN_Model3(nn.Module):
    def __init__(self):
        super(DNN_Model3, self).__init__()
        self.hidden1 = nn.Linear(1, 190)   
        self.hidden2 = nn.Linear(190, 1)

    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = self.hidden2(x)
        return x

def train_model(model, x, y, num_epochs=1000, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_list = []
    minimal_ratio_list = []
    grad_list = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        
        # Record loss
        loss_list.append(loss.item())

        grad_all = 0.0
        for p in model.parameters():
            grad = 0.0
            if p.grad is not None:
                grad = (p.grad.cpu().data.numpy()**2).sum()
            grad_all += grad
        grad_norm = grad_all ** 0.5
        
        grad_list.append(grad_norm)

        # Check gradient norm
        if grad_norm < 1e-3:
            print(f"Gradient norm close to zero at epoch {epoch}")
            break

        optimizer.step()
    
    # Calculate minimal ratio using the provided function
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eigenvalues, _ = get_hessian_eigenvectors(model, criterion, [(x, y)], 1, 'cpu', num_params-1, None)
    minimal_ratio = (eigenvalues > 0).sum().item() / eigenvalues.size
    minimal_ratio_list.append(minimal_ratio)

    return loss_list, minimal_ratio_list

# Train the model multiple times
num_trains = 100
loss_history = []
minimal_ratio_history = []

for _ in range(num_trains):
    model = DNN_Model3()

    loss_list, minimal_ratio_list = train_model(model, x, y, num_epochs=1000, lr=0.01)
    loss_history.append(loss_list[-1])
    minimal_ratio_history.append(minimal_ratio_list[-1])

# Plotting
plt.figure(figsize=(10, 5))
for i in range(num_trains):
    plt.scatter(minimal_ratio_history[i], loss_history[i], alpha=0.3, color='blue')

plt.ylabel('Loss')
plt.xlabel('Minimal Ratio')
plt.title('Minimal Ratio vs Loss')
plt.grid(True)
plt.show()
