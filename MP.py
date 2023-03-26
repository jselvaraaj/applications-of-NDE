import torch
import copy

class DegeneratedMarkovProcess:

    def __init__(self,env,policy,device=None) -> None:
        self.env = copy.deepcopy(env)
        self.policy = policy

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

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

        