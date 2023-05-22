import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import math
from scipy.integrate import quad_vec

class TrigExpMDP(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode=None, size=5):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        
        self.name = "TrigExpMDP"

        
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = spaces.Box(0, 1.0)
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.history = []


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
      self._agent_location = (self.np_random.integers(0, self.size, size=2, dtype=int)).astype(np.float32)

      self.history = [self._agent_location]

      observation = self._get_obs()
      info = self._get_info()

      if self.render_mode == "human":
          self._render_frame()

      return observation, info

    def step(self, action):

      fun1 = lambda a: np.asarray([math.exp(math.cos(a)**2), math.exp(math.sin(a)**2)])
      
      self._agent_location = (self._agent_location + quad_vec(fun1,0,action)[0]).astype(np.float32)
    
      observation = self._get_obs()
      info = self._get_info()

      self.history.append(self._agent_location)

      if self.render_mode == "human":
          self._render_frame()

      return observation, 0, False, False, info

    def render(self):
      if self.render_mode == "rgb_array":
          return self._render_frame()

    def _render_frame(self):
      if self.window is None and self.render_mode == "human":
          pygame.init()
          pygame.display.init()
          self.window = pygame.display.set_mode(
              (self.window_size, self.window_size)
          )
      if self.clock is None and self.render_mode == "human":
          self.clock = pygame.time.Clock()

      canvas = pygame.Surface((self.window_size, self.window_size))

      canvas.fill((255, 255, 255))
      pix_square_size = (
          self.window_size / self.size
      )  # The size of a single grid square in pixels

      temp = pygame.Surface((pix_square_size, pix_square_size))

      #history

      if len(self.history) >= 2:
        last = self.history[0]
        for past_coord in self.history[1:]:
          pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * past_coord,
                (pix_square_size, pix_square_size),
            ),
          )

          pygame.draw.line(
              canvas,
              1,
              (last + 0.5) * pix_square_size,
              (past_coord + 0.5) * pix_square_size,
              width=1,
          )


          last = past_coord

      
      # Now we draw the agent
      pygame.draw.circle(
          canvas,
          (0, 0, 255),
          (self._agent_location + 0.5) * pix_square_size,
          pix_square_size / 3,
      )

      

      # Finally, add some gridlines
      for x in range(self.size + 1):
          pygame.draw.line(
              canvas,
              0,
              (0, pix_square_size * x),
              (self.window_size, pix_square_size * x),
              width=3,
          )
          pygame.draw.line(
              canvas,
              0,
              (pix_square_size * x, 0),
              (pix_square_size * x, self.window_size),
              width=3,
          )

      if self.render_mode == "human":
          # The following line copies our drawings from `canvas` to the visible window
          self.window.blit(canvas, canvas.get_rect())
          pygame.event.pump()
          pygame.display.update()

          # We need to ensure that human-rendering occurs at the predefined framerate.
          # The following line will automatically add a delay to keep the framerate stable.
          self.clock.tick(self.metadata["render_fps"])
      else:  # rgb_array
          return np.transpose(
              np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
          )

    def close(self):
      if self.window is not None:
          pygame.display.quit()
          pygame.quit()
