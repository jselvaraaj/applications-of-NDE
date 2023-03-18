import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchcde
import torchdiffeq


class DegeneratedMarkovStateEvolver(torch.nn.Module):
    def __init__(self, state_dim,hidden_channels):
        super(DegeneratedMarkovStateEvolver, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_channels),
            nn.Tanh(),
            nn.Linear(hidden_channels, state_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)
    
class DynamicsTrajectoryDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):

        super(DynamicsTrajectoryDE, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)
      

    def forward(self, t, z):
        z = self.linear1(z)
        z = z.relu()

        z = self.linear2(z)
        z = z.tanh()

        z = z.view(z.size(0), self.hidden_channels, self.input_channels)

        return z



class DynamicsFunction(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic", evolve = 10,device = None):
        super(DynamicsFunction, self).__init__()

        self.initial = torch.nn.Linear(input_channels, hidden_channels)

        self.DEfunc = DynamicsTrajectoryDE(input_channels, hidden_channels)

        self.readout = torch.nn.Linear(hidden_channels, output_channels)

        self.interpolation = interpolation

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.evolve = torch.tensor(float(evolve)).to(device)

        self.evolver = DegeneratedMarkovStateEvolver(hidden_channels,hidden_channels)

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        X0 = X.evaluate(X.interval[0])

    
        z0 = self.initial(X0)

        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.DEfunc,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = z_T
        
        n = pred_y.shape[0]
        
        pred_y = torchdiffeq.odeint(self.evolver, pred_y, torch.stack((torch.tensor(0.0),self.evolve)).to(self.device))
        
        pred_y = self.readout(pred_y)

        return pred_y[1]


class NNBaseline(torch.nn.Module):
    def __init__(self, state_dim,hidden_channels,evolve = 10,device = None):
        super(NNBaseline, self).__init__()

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

        self.evolve = evolve 

        self.net.to(self.device)

        self.rnn = nn.RNN(state_dim, hidden_channels, 1, batch_first=True)

        self.rnn.float()

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, y):
        
        # y = y[:,-1,:]
        
        output, hn = self.rnn(y)

        y = hn[-1]
        for i in range(self.evolve):
            y = self.net(y)

        y = self.readout(y)

        return y
    
# class Encoder(torch.nn.Module):
#     def __init__(self, state_dim,hidden_channels,output_channels,device = None):
#         super(NNBaseline, self).__init__()

#         self.net = nn.Sequential(
#             nn.Linear(state_dim, hidden_channels),
#             nn.Tanh(),
#             nn.Linear(hidden_channels, output_channels),
#         )

#         if device is None:
#             self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         else:
#             self.device = device

#         self.net.to(self.device)

#         for m in self.net.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.constant_(m.bias, val=0)

#     def forward(self, x):
#         return self.net(x)