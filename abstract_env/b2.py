import math

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class B2Env(gym.Env):
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
        p, v = self.state
        done = False
        # # u = np.clip(u, -self.th, self.th)[0]
        # offset = 0
        # scala = 1
        # offset = 0
        # scala = 2
        # # print(u[0])
        # u = u[0] - offset
        # u = scala * u
        # # print(u)
        # # u = u * 4
        # # t = 0.02
        # t = 0.2
        # p_new = p + (v - p * p * p) * t
        # v_new = v + u * t
        #
        # self.state = np.array([p_new, v_new], dtype=np.float32)

        # reward = -2
        self.step_size(u)
        p_new, v_new = self.state
        reward = -math.fabs(p_new) - math.fabs(v_new + 0.3)

        if 0.03 >= p_new >= -0.02 and -0.2 >= v_new >= -0.33:
            reward = 100
            done = True

        # done = bool(
        #     abs(p_new) > 1.5 or
        #     abs(v_new) > 1.5
        # ) or done
        #
        # if bool(
        #         abs(p_new) > 1.5 or
        #         abs(v_new) > 1.5
        # ):
        #     reward = -600

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.9, 0.9])
        low = np.array([0.7, 0.7])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.001):
        done = False
        offset = 0
        scala = 1
        u = u[0] - offset
        u = scala * u
        t = 0.2
        time = 0
        state_list = []
        while time <= t:
            p, v = self.state
            p_new = p + (v - p * p * p) * step_size
            v_new = v + u * step_size
            self.state = np.array([p_new, v_new], dtype=np.float32)
            state_list.append([p_new, v_new])
            if 0.1 >= p_new >= -0.3 and 0.5 >= v_new >= -0.35:
                # reward = 0
                done = True
            time = round(time + step_size, 10)
        return self._get_obs(), state_list, done

    # def angle_normalize(self, x):
    #     return (( (x + np.pi) % (2 * np.pi) ) - np.pi)
