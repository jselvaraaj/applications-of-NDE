import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchcde
import torchdiffeq
import utils


class DegeneratedMarkovStateEvolver(torch.nn.Module):
    def __init__(self, input_channels,hidden_channels):
        super(DegeneratedMarkovStateEvolver, self).__init__()

        self.device = utils.get_device()

        self.net = nn.Sequential(
            nn.Linear(input_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, input_channels),
        ).to(self.device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

class DynamicsFunction(torch.nn.Module):
    def __init__(self, state_dim, hidden_channels,observation_space_dim):
        super(DynamicsFunction, self).__init__()
        
        self.device = utils.get_device()

        self.readin = nn.Sequential(
            nn.Linear(state_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        ).to(self.device)

        self.evolver = DegeneratedMarkovStateEvolver(hidden_channels,hidden_channels).to(self.device)

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, observation_space_dim),
        ).to(self.device)


    def forward(self, x, evolve_len):
        x = self.readin(x)
        evolve_len = torch.tensor(float(evolve_len))
        t = torch.stack((torch.tensor(0.0),evolve_len)).to(self.device)
        pred_y = torchdiffeq.odeint(self.evolver, x, t)
        pred_y = self.readout(pred_y.to(self.device))
        return pred_y[1]


class NNBaseline(torch.nn.Module):
    def __init__(self, state_dim,hidden_channels,observation_space_dim):
        super(NNBaseline, self).__init__()

        self.device = utils.get_device()

        self.readin = nn.Sequential(
            nn.Linear(state_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        ).to(self.device)

        self.net = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, hidden_channels),
        ).to(self.device)

        self.readout = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, observation_space_dim),
        ).to(self.device)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, x,evolove_len):
        hidden_representation = self.readin(x)
        for i in range(evolove_len):
            hidden_representation = self.net(hidden_representation)
        x = self.readout(hidden_representation)
        return x
  