import torch
import copy
import utils

class DegeneratedMarkovProcess:

    def __init__(self,env,policy) -> None:
        self.env = copy.deepcopy(env)
        self.policy = policy
        self.device = utils.get_device()
        self.observation = None
        self.reset()

    def reset(self):
        self.env.reset()
        self.observation = torch.from_numpy(self.env._get_obs()[0]).to(self.device)

    def get_obs(self):
        return torch.hstack((self.observation,self.policy.weights.to(self.device)))

    def step(self):
        action = self.policy.get_action(self.observation)

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.observation = torch.from_numpy(obs[0]).to(self.device)

        return terminated

        