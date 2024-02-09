import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define DNN model
class DNN_Model(nn.Module):
    def __init__(self):
        super(DNN_Model, self).__init__()
        self.hidden1 = nn.Linear(28*28, 64)
        self.hidden2 = nn.Linear(64, 32)
        self.hidden3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x

def train_model(model, optimizer, epochs):
    model.train()
    all_epoch_loss = []
    all_accuracy = []
    all_weight_df = pd.DataFrame()

    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_samples = 0

        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            prediction = model(inputs)

            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(prediction, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += labels.size(0)
            _, predicted_classes = torch.max(prediction.data, 1)
            correct_predictions += (predicted_classes == labels).sum().item()

        epoch_loss = total_loss / len(train_loader)
        accuracy=correct_predictions/total_samples

        all_epoch_loss.append(epoch_loss)
        all_accuracy.append(accuracy)

        print(f'Epoch [{epoch}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

        # collect weights
        weight_df = pd.DataFrame()
        for name, parameter in model.named_parameters():
            if 'weight' in name:
                ws=torch.nn.utils.parameters_to_vector(parameter).detach().numpy()
                weight_df = pd.concat([weight_df, pd.DataFrame(ws).T], axis = 1)

        all_weight_df = pd.concat([all_weight_df, weight_df], axis = 0)

    return all_epoch_loss, all_accuracy, all_weight_df

total_loss = []
total_epoch_loss = []
total_accuracy = []
total_weight_df=pd.DataFrame()


for i in range(8):
    print('Times: ', i)
    model = DNN_Model()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    all_epoch_loss, all_accuracy, all_weight_df = train_model(model, optimizer, 60)
    print(f'Loss: {all_epoch_loss:}, Accuracy: {all_accuracy:}')
    total_accuracy.append(all_accuracy)
    total_epoch_loss.append(all_epoch_loss)
    total_weight_df = pd.concat([total_weight_df, all_weight_df], axis = 0)

weight_array = np.array(total_weight_df)
pca = PCA(n_components=2).fit_transform(weight_array)
pca_df = pd.DataFrame(pca, columns=['x','y'])
total_accuracy_list = [item for sublist in total_accuracy for item in sublist]
pca_df['accuracy'] = total_accuracy_list
pca_selected_df = pca_df.iloc[::3, :]

for i in range(pca_selected_df.shape[0]):
    x = pca_selected_df['x'][i*3]
    y = pca_selected_df['y'][i*3]
    accuracy = pca_selected_df['accuracy'][i*3]
    plt.scatter(x, y, marker = f'${accuracy}$')

plt.title("Whole Model - PCA")
plt.show()

first_layer_weights_array = np.array(total_weight_df.iloc[:, 0:28*28])
pca_layer1 = PCA(n_components=2).fit_transform(first_layer_weights_array)
pca_layer1_df = pd.DataFrame(pca_layer1,columns=['x','y'])
pca_layer1_df['accuracy'] = total_accuracy_list
pca_layer1_selected_df = pca_layer1_df.iloc[::3, :]
pca_layer1_selected_df

for i in range(pca_layer1_selected_df.shape[0]):
    x = pca_layer1_selected_df['x'][i*3]
    y = pca_layer1_selected_df['y'][i*3]
    accuracy = pca_layer1_selected_df['accuracy'][i*3]
    plt.scatter(x, y, marker = f'${accuracy}$')

plt.title("Layer 1 - PCA")
plt.show()