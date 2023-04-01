import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchcde
import torchdiffeq


class DegeneratedMarkovStateEvolver(torch.nn.Module):
    def __init__(self, input_channels,hidden_channels):
        super(DegeneratedMarkovStateEvolver, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, input_channels),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class DynamicsFunction(torch.nn.Module):
    def __init__(self, state_dim, hidden_channels,device = None):
        super(DynamicsFunction, self).__init__()

        self.initial = nn.Sequential(
            nn.Linear(state_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        )

        self.evolver = DegeneratedMarkovStateEvolver(hidden_channels,hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, state_dim)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    def forward(self, x, evolve_len):

        x = self.initial(x)
        evolve_len = torch.tensor(float(evolve_len))
        
        pred_y = torchdiffeq.odeint(self.evolver, x, torch.stack((torch.tensor(0.0),evolve_len)).to(self.device))
        pred_y = self.readout(pred_y)
        return pred_y[1]


class NNBaseline(torch.nn.Module):
    def __init__(self, state_dim,hidden_channels,device = None):
        super(NNBaseline, self).__init__()

        self.initial = torch.nn.Linear(state_dim, hidden_channels)
        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.readout = torch.nn.Linear(hidden_channels, state_dim)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.net.to(self.device)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x,evolove_len):
        
        hidden_representation = self.initial(x)
        for i in range(evolove_len):
            hidden_representation = self.net(hidden_representation)
        x = self.readout(hidden_representation)

        return x
  