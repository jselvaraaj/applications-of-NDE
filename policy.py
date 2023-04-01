import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy:

    def __init__(self, state_space_size,action_space_size,device= None) -> None:
        self.policy = PolicyNetwork(state_space_size,action_space_size)
        self.input_size = state_space_size
        self.output_size = action_space_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.policy.to(device)

        self._update_weights()

    def get_action(self,obs):
        return torch.argmax(self.policy(obs[None,:].to(torch.float))).cpu().detach().item()

    def _update_weights(self):
        policy_weights = []

        for _, param in self.policy.named_parameters():
            policy_weights.append(torch.flatten(param.detach().clone()))

        policy_weights = torch.hstack(policy_weights)

        self.weights = policy_weights


class PolicyNetwork(nn.Module):

    def __init__(self,state_space_size,action_space_size):
        super(PolicyNetwork, self).__init__()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.net = nn.Sequential(
            nn.Linear(state_space_size, 8),
            nn.Tanh(),
            nn.Linear(8, action_space_size),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, x):
        return self.net(x)
