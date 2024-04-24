import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.parametrize import register_parametrization


#%%

def positive_part(x):
    return torch.maximum(x, torch.zeros_like(x))


class SoftplusParameterization(nn.Module):
    def forward(self, X):
        return nn.functional.softplus(X)

#%%

class FICNN(nn.Module):
    def __init__(self, in_dim = 2, out_dim = 1, n_layers = 4, hidden_dim = 8, activation_fn = nn.ReLU()):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.activation_fn = activation_fn
        self.loss_fn = nn.BCELoss()
        
        Wz = []
        Wy = []
        Wy.append(nn.Linear(in_dim, hidden_dim))
        
        for _ in range(n_layers - 2):
            Wz.append(nn.Linear(hidden_dim, hidden_dim, bias = False))
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            Wy.append(nn.Linear(in_dim, hidden_dim))
        
        Wz.append(nn.Linear(hidden_dim, out_dim, bias = False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wy.append(nn.Linear(in_dim, out_dim))
        
        self.Wz = nn.ModuleList(Wz)
        self.Wy = nn.ModuleList(Wy)
    
    
    def forward(self, y, x = None):
        z = self.Wy[0](y)
        for (layer_y, layer_z) in zip(self.Wy[1:self.n_layers - 1], self.Wz):
            z = self.activation_fn(layer_z(z) + layer_y(y))
        
        z = self.Wz[-1](z) + self.Wy[-1](y)
        
        
        return nn.Sigmoid()(z.squeeze(-1))
    

#%%



class PICNN(nn.Module):
    def __init__(self, x_dim = 2, y_dim = 2, n_layers = 4, u_dim = 8, z_dim = 8,
                 activation_fn_u = nn.ReLU(), activation_fn_z = nn.ReLU()):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.n_layers = n_layers
        self.u_dim = u_dim
        self.z_dim = z_dim
        self.activation_fn_u = activation_fn_u
        self.activation_fn_z = activation_fn_z
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        Wbar = []
        Wz = []
        Wzu = []
        Wy = []
        Wyu = []
        Wu = []
        
        Wbar.append(nn.Linear(x_dim, u_dim))
        Wz.append(nn.Linear(y_dim, z_dim, bias = False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wzu.append(nn.Linear(x_dim, y_dim))
        Wy.append(nn.Linear(y_dim, z_dim, bias = False))
        Wyu.append(nn.Linear(x_dim, y_dim))
        Wu.append(nn.Linear(x_dim, z_dim))
        
        for _ in range(n_layers - 2):
            Wbar.append(nn.Linear(u_dim, u_dim))
            Wz.append(nn.Linear(z_dim, z_dim, bias = False))
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            Wzu.append(nn.Linear(u_dim, z_dim))
            Wy.append(nn.Linear(y_dim, z_dim, bias = False))
            Wyu.append(nn.Linear(u_dim, y_dim))
            Wu.append(nn.Linear(u_dim, z_dim))
            
        
        Wz.append(nn.Linear(z_dim, z_dim, bias = False))
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        Wzu.append(nn.Linear(u_dim, z_dim))
        Wy.append(nn.Linear(y_dim, z_dim, bias = False))
        Wyu.append(nn.Linear(u_dim, y_dim))
        Wu.append(nn.Linear(u_dim, z_dim))
        
        
        self.Wbar = nn.ModuleList(Wbar)
        self.Wz = nn.ModuleList(Wz)
        self.Wzu = nn.ModuleList(Wzu)
        self.Wy = nn.ModuleList(Wy)
        self.Wyu = nn.ModuleList(Wyu)
        self.Wu = nn.ModuleList(Wu)
        
    
    def forward(self, x, y):
        
        u = x
        z = y
        
        for i in range(self.n_layers - 1):
            z = self.activation_fn_z(self.Wz[i](z * positive_part(self.Wzu[i](u))) +\
                                     self.Wy[i](y * self.Wyu[i](u)) +\
                                     self.Wu[i](u))
            u = self.activation_fn_u(self.Wbar[i](u))
        
        
        z = self.Wz[i](z * positive_part(self.Wzu[i](u))) +\
            self.Wy[i](y * self.Wyu[i](u)) +\
            self.Wu[i](u)
        
        return z
    
    
#%%

class PICNN_Conv2d(nn.Module):
    def __init__(
            self,
            nr_channels,        # tuple of channels
            kernel_sizes,       # tuple of kernel sizes
            strides,            # tuple of strides
            in_channels = 1,
            activation_fn = nn.ReLU(),
    ):
        super().__init__()
        self.nr_channels = nr_channels
        self.kernel_size = kernel_sizes
        self.strides = strides
        self.activation_fn = activation_fn
        bn = [nn.BatchNorm2d(num_features = nr_ch) for nr_ch in nr_channels]
        self.bn = nn.ModuleList(bn)
        self.nr_layers = len(nr_channels)
        
        Wbar = []
        Wz = []
        Wzu = []
        Wy = []
        Wyu = []
        Wu = []
        
        layer = 0
        Wbar.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], stride = strides[layer]))
        Wzu.append(nn.Conv2d(in_channels, in_channels, 3, padding = "same"))
        Wz.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], 
                            stride = strides[layer], bias = False))
        Wyu.append(nn.Conv2d(in_channels, 1, 3, padding = "same"))
        Wy.append(nn.Conv2d(1, nr_channels[layer], kernel_sizes[layer], stride = strides[layer]))
        Wu.append(nn.Conv2d(in_channels, nr_channels[layer], kernel_sizes[layer], stride = strides[layer]))
        layer += 1
        
        register_parametrization(Wz[-1], "weight", SoftplusParameterization())
        
        while layer < self.nr_layers:
            Wbar.append(nn.Conv2d(nr_channels[layer - 1], nr_channels[layer], kernel_sizes[layer], 
                                  stride = strides[layer]))
            Wzu.append(nn.Conv2d(nr_channels[layer - 1], nr_channels[layer - 1], 3, padding = "same"))
            Wz.append(nn.Conv2d(nr_channels[layer - 1], nr_channels[layer], kernel_sizes[layer], 
                                stride = strides[layer], bias = False))
            Wyu.append(nn.Conv2d(nr_channels[layer - 1], 1, 3, padding = "same"))
            Wy.append(nn.Conv2d(1, nr_channels[layer], kernel_sizes[layer], stride = strides[layer]))
            Wu.append(nn.Conv2d(nr_channels[layer - 1], nr_channels[layer], kernel_sizes[layer], 
                                stride = strides[layer]))
            
            register_parametrization(Wz[-1], "weight", SoftplusParameterization())
            layer += 1
        
        
        self.Wbar = nn.ModuleList(Wbar)
        self.Wz = nn.ModuleList(Wz)
        self.Wzu = nn.ModuleList(Wzu)
        self.Wy = nn.ModuleList(Wy)
        self.Wyu = nn.ModuleList(Wyu)
        self.Wu = nn.ModuleList(Wu)
        
        
        self.fc_uz = nn.Linear(768, 768)
        self.fc_z = nn.Linear(768, 1, bias = False)
        self.fc_u = nn.Linear(768, 1)
        
        register_parametrization(self.fc_z, "weight", SoftplusParameterization())
        
        
    def forward(self, x, y):
        u = x
        z = y
        us = [x]
        zs = [y]
        
        for i in range(self.nr_layers):
            z = self.activation_fn(
                self.Wz[i](z * positive_part(self.Wzu[i](u))) +\
                self.Wy[i](F.interpolate(y, size = self.Wyu[i](u).shape[2:]) * self.Wyu[i](u)) + self.Wu[i](u)
            )
            u = self.bn[i](self.activation_fn(self.Wbar[i](u)))
        
        u = u.view(-1, 768)
        z = z.view(-1, 768)
        
        z = self.activation_fn(self.fc_z(positive_part(self.fc_uz(u)) * z) + self.fc_u(u))
        
        return z

 
    
    
    
    