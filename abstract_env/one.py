import math

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class OneEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([5.0, 1.0], dtype=np.float32)
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
        y, x = self.state
        done = False
        # u = np.clip(u, -self.th, self.th)[0]
        # offset = 0.1
        # offset = 0
        # scala = 1
        # u = u[0] - offset
        # u = scala * u
        # # u = np.clip(u, -2, 2)
        # t = 0.1
        # p_new = p + v * t
        # v_new = v + (u * v * v - p) * t
        #
        # self.state = np.array([p_new, v_new], dtype=np.float32)
        self.step_size(u)
        y, x = self._get_obs()

        reward = -abs(y - (1 / (1 - x)))

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([1, 0])
        low = np.array([1, 0])

        self.state = self.np_random.uniform(low=low, high=high)

        # self.state = np.array([0.85, 0.55])

        return self._get_obs()

    def _get_obs(self):
        # self.state[0] = np.clip(self.state[0], -2, 2)
        # self.state[1] = np.clip(self.state[1], -2, 2)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.0001):
        t = 0.01
        time = 0
        state_list = []
        done = False
        offset = 0
        scala = 10
        u = u[0] - offset
        u = scala * u
        while time <= t:
            y, x = self.state
            # u = np.clip(u, -self.th, self.th)[0]
            # offset = 0.1
            y_new = y + u * step_size
            x_new = x + step_size
            # if math.isnan(v_new):
            #     print("v")
            y_new = np.clip(y_new,-15,15)

            self.state = np.array([y_new, x_new], dtype=np.float32)
            state_list.append([y_new, x_new])
            # if 0.15 >= p_new >= 0.02 and 0.28 >= v_new >= 0.07:
            #     done = True
            time = round(time + step_size, 10)
        return self.state, state_list, done

# def angle_normalize(self, x):
#     return (( (x + np.pi) % (2 * np.pi) ) - np.pi)
