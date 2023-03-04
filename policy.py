import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy:

    def __init__(self, state_space_size,action_space_size) -> None:
        self.policy = PolicyNetwork(state_space_size,action_space_size)
        self.input_size = state_space_size
        self.output_size = action_space_size

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

        self.fc1 = nn.Linear(state_space_size, 64)  
        self.fc2 = nn.Linear(64, action_space_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.softmax(x)
        return x
