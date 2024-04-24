import torch
import torch.nn as nn
from ICNN_architectures import FICNN
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm

#%% Approximate function with support in the unit cube

def f(x):
    return torch.sum(x * x, dim = -1)

#%% Synthetic data

nr_samples = 10000
dim = 2
X = torch.rand((nr_samples, dim)) * 2 - 1
y = f(X)

dataset = torch.utils.data.TensorDataset(X, y)

#%% Model training

model = FICNN(in_dim = 2, out_dim = 1, n_layers = 4, hidden_dim = 8)

# Parameters for training
epochs = 500
batch_size = 200
lr = 0.001

# Dataset
train_loader = DataLoader(dataset, batch_size, shuffle = True)

# Training
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
model.train()
for i in range(epochs):
    epoch_avg_loss = 0
    for (data, target) in train_loader:
        predicted = model(data)
        loss = loss_fn(predicted, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_avg_loss += loss.item() * batch_size / len(dataset)
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(i+1, epochs, epoch_avg_loss))


#%% Plotting

xg = np.linspace(-1, 1, 1000)
yg = np.linspace(-1, 1, 1000)
X = np.dstack(np.meshgrid(xg, yg))

xgrid, ygrid = np.meshgrid(xg, yg)
zg = np.abs(model(torch.Tensor(X)).detach().numpy() - f(torch.Tensor(X)).numpy())

fig, ax = plt.subplots()
contourf_ = ax.contourf(xg, yg, zg, 100)
cbar = fig.colorbar(contourf_)
plt.show()











