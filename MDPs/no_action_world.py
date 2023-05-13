import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

class NoActionMDP(gym.Env):
    def __init__(self, size=5,delta=0.5):
        self.size = size 
        self.name = "NoActionMDP"


        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )
        
        self.delta = delta


    def _get_obs(self):
      return self._agent_location.astype(np.float32)

    def _get_info(self):
      return {
        "distance": np.linalg.norm(
            self._agent_location, ord=1
          )
        }

    def reset(self, seed=None, options=None):
      # We need the following line to seed self.np_random
      super().reset(seed=seed)

      # Choose the agent's location uniformly at random
      self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int).astype(np.float32)

      observation = self._get_obs()
      info = self._get_info()

      return observation, info

    def step(self, action = None):        
      self._agent_location = (self._agent_location + self.delta **2).astype(np.float32)
        
      observation = self._get_obs()
      info = self._get_info()

      return observation, 0, False, False, info

    def render(self):
      pass

    def close(self):
      pass
