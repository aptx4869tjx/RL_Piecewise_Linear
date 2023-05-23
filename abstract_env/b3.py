import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path


class B3Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2], dtype=np.float32)
        self.action_space = spaces.Box(
            low=-self.th,
            high=self.th,
            shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x1, x2 = self.state
        done = False

        # self.state = np.array([x1_new, x2_new], dtype=np.float32)
        self.step_size(u)
        x1_new, x2_new = self.state
        reward = -abs(x1_new - 0.25) - abs(x2_new + 0.06)
        # reward = -1
        #
        # if 0.3 >= x1_new >= 0.2 and -0.05 >= x2_new >= -0.3:
        #     reward = 200
        #     done = True
        if 0.29 >= x1_new >= 0.2 and -0.05 >= x2_new >= -0.12:
            reward = 200
            done = True

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.9, 0.5])
        low = np.array([0.8, 0.4])

        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.002):

        done = False
        offset = 0
        scala = 1
        u = u[0] - offset
        u = scala * u
        t = 0.1
        time = 0
        state_list = []
        while time <= t:
            x1, x2 = self.state
            x1_new = x1 - x1 * (0.1 + (x1 + x2) * (x1 + x2)) * step_size
            x2_new = x2 + (u + x1) * (0.1 + (x1 + x2) * (x1 + x2)) * step_size
            state_list.append([x1_new, x2_new])
            self.state = np.array([x1_new, x2_new], dtype=np.float32)

            if 0.3 >= x1_new >= 0.2 and -0.05 >= x2_new >= -0.3:
                done = True
            time = round(time + step_size, 10)
        return self.state, state_list, done
