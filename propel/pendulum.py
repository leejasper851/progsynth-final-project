import gymnasium as gym
import numpy as np

class PendulumThetaEnv(gym.Env):
    def __init__(self, pendulum_env):
        self.pendulum_env = pendulum_env
    
    def _thetafy(self, observation):
        theta = np.arctan2(observation[1], observation[0])
        if theta < 0:
            theta += 2 * np.pi
        return np.array([theta, observation[2]], dtype=np.float32)
    
    def reset(self):
        observation, info = self.pendulum_env.reset()
        return self._thetafy(observation), info
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.pendulum_env.step(action)
        return self._thetafy(observation), reward, terminated, truncated, info
    
    def render(self):
        return self.pendulum_env.render()

    def close(self):
        self.pendulum_env.close()
