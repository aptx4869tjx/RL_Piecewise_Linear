import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from os import path


class ToraEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2, 3.5, 2], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-2]),
            high=np.array([2]),
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
        x1, x2, x3, x4 = self.state
        done = False

        # u1 = np.clip(u, -10, 10)[0]
        # offset = 0
        # scala = 1
        # u1 = u[0] - offset
        # u1 = u1 * scala
        #
        # t = 0.02
        #
        # x1_new = x1 + x2 * t
        # x2_new = x2 + (-x1 + 0.1 * math.sin(x3)) * t
        # x3_new = x3 + x4 * t
        # x4_new = x4 + u1 * t
        #
        # self.state = np.array([x1_new, x2_new, x3_new, x4_new], dtype=np.float32)

        # reward = -1
        self.step_size(u)
        x1_new, x2_new, x3_new, x4_new = self.state
        # if abs(x1_new) > 2 or abs(x2_new) > 2 or abs(x4_new) > 2:
        #     reward = -800
        #     done = True

        reward = - (1 * abs(x1_new - 0.05) + 1 * abs(x2_new + 0.7))
        # reward = -  2 * abs(x2_new + 0.7)
        # reward = -1
        # if 0.1 >= x1_new >= -0.1 and -0.6 >= x2_new >= -0.8:
        #     reward = 1000
        #     done = True
        if 0.18 >= x1_new >= -0.1 and -0.6 >= x2_new >= -0.9:
            reward = 1000
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
        high = np.array([-0.75, -0.43, 0.54, -0.28])
        low = np.array([-0.77, -0.45, 0.51, -0.3])
        # high = np.array([-0.766, -0.448, 0.54, -0.28])
        # low = np.array([-0.767, -0.449, 0.53, -0.3])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        self.state[0] = np.clip(self.state[0], -2, 2)
        self.state[1] = np.clip(self.state[1], -2, 2)
        # self.state[2] = np.clip(self.state[2], -math.pi / 2, math.pi / 2)
        # self.state[3] = np.clip(self.state[3], -2, 2)
        self.state[2] = np.clip(self.state[2], -20, 20)
        self.state[3] = np.clip(self.state[3], -10, 10)

        return self.state
        # return np.array(transition([np.cos(theta), np.sin(theta), thetadot]))

    def step_size(self, u, step_size=0.005):

        done = False

        offset = 0
        scala = 1
        u1 = u[0] - offset
        u1 = u1 * scala

        t = 0.5
        time = 0
        state_list = []
        while time <= t:
            x1, x2, x3, x4 = self.state
            x1_new = x1 + x2 * step_size
            x2_new = x2 + (-x1 + 0.1 * math.sin(x3)) * step_size
            x3_new = x3 + x4 * step_size
            x4_new = x4 + u1 * step_size
            state_list.append([x1_new, x2_new])
            self.state = np.array([x1_new, x2_new, x3_new, x4_new], dtype=np.float32)

            if 0.18 >= x1_new >= -0.1 and -0.6 >= x2_new >= -0.9:
                done = True
                break
            time = round(time + step_size, 10)
        return self.state, state_list, done

# def angle_normalize(self, x):
#     return (( (x + np.pi) % (2 * np.pi) ) - np.pi)
