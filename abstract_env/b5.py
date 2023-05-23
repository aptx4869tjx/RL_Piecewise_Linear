import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path


class B5Env(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2, 2], dtype=np.float32)
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
        x1, x2, x3 = self.state
        done = False
        # u = np.clip(u, -self.th, self.th)[0]
        #
        # offset = 0
        # scala = 1
        # u = u[0] - offset
        # u = u * scala
        #
        # t = 0.05
        # # t=0.1
        # x1_new = x1 + (x1 * x1 * x1 - x2) * t
        # x2_new = x2 + x3 * t
        # x3_new = x3 + u * t
        self.step_size(u)
        # self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)
        x1_new, x2_new, x3_new = self.state

        reward = -abs(x1_new + 0.34) - abs(x2_new - 0.1)

        # if -0.3 >= x1_new >= -0.38 and 0.22 >= x2_new >= 0.05:
        #     reward = 500
        #     done = True

        if -0.29 >= x1_new >= -0.38 and 0.21 >= x2_new >= 0.05:
            reward = 500
            done = True

        # done = bool(
        #     abs(x1_new) > 1.5 or
        #     abs(x2_new) > 1.5
        # ) or done
        #
        # if bool(
        #         abs(x1_new) > 1.5 or
        #         abs(x2_new) > 1.5
        # ):
        #     reward = -600

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.4, 0.47, 0.27])
        low = np.array([0.38, 0.45, 0.25])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        self.state[2] = np.clip(self.state[2], -5, 5)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.01):

        done = False
        # u = np.clip(u, -self.th, self.th)[0]

        offset = 0
        scala = 1
        u = u[0] - offset
        u = u * scala
        t = 0.2
        time = 0
        state_list = []
        while time <= t:
            x1, x2, x3 = self.state
            x1_new = x1 + (x3 * x3 * x3 - x2) * step_size
            x2_new = x2 + x3 * step_size
            x3_new = x3 + u * step_size
            state_list.append([x1_new, x2_new])
            self.state = np.array([x1_new, x2_new, x3_new], dtype=np.float32)

            if -0.28 >= x1_new >= -0.4 and 0.22 >= x2_new >= 0.05:
                done = True
            time = round(time + step_size, 10)
            if done:
                break
        return self.state, state_list, done
