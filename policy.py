import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class Policy:

    def __init__(self, state_space_size,action_space_size) -> None:
        self.policy = PolicyNetwork(state_space_size,action_space_size)
        self.input_size = state_space_size
        self.output_size = action_space_size

        self.device = utils.get_device()

        self.policy.to(self.device)

        self._update_weights()

    def get_action(self,obs):
        # return torch.argmax(self.policy(obs[None,:].to(torch.float))).cpu().detach().item()
        return torch.argmax(self.policy(obs[None,:].to(torch.float))).detach().item()

    def _update_weights(self):
        policy_weights = []

        for _, param in self.policy.named_parameters():
            policy_weights.append(torch.flatten(param.detach().clone()))

        policy_weights = torch.hstack(policy_weights)

        self.weights = policy_weights


class PolicyNetwork(nn.Module):

    def __init__(self,state_space_size,action_space_size):
        super(PolicyNetwork, self).__init__()
        
        self.device = utils.get_device()

        self.net = nn.Sequential(
            nn.Linear(state_space_size, 4),
            # nn.Tanh(),
            nn.Linear(4, action_space_size),
            nn.Softmax(dim=1)
        ).to(self.device)

    def forward(self, x):
        return self.net(x)
