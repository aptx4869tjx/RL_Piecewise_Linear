import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math


class QuadEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self):
        self.th = 1.0
        # self.viewer = None

        high = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-2, -2, -2]),
            high=np.array([2, 2, 2]),
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
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12 = self.state
        done = False

        # u1 = np.clip(u, -10, 10)[0]
        offset = 0
        scala = 1
        u0 = u[0] - offset
        u0 = u0 * scala
        u1 = u[1] - offset
        u1 = u1 * scala
        u2 = u[2] - offset
        u2 = u2 * scala

        t = 0.05
        small_t = 0.005
        time = 0
        reward = 0
        bias = 0
        while time < t:
            x1 = x1 + (math.cos(x8) * math.cos(x9) * x4 + (
                    math.sin(x7) * math.sin(x8) * math.cos(x9) - math.cos(x7) * math.sin(x9)) * x5 + (
                               math.cos(x7) * math.sin(x8) * math.cos(x9) + math.sin(x7) * math.sin(x9)) * x6) * small_t
            x2 = x2 + (math.cos(x8) * math.sin(x9) * x4 + (
                    math.sin(x7) * math.sin(x8) * math.sin(x9) + math.cos(x7) * math.cos(x9)) * x5 + (
                               math.cos(x7) * math.sin(x8) * math.sin(x9) - math.sin(x7) * math.cos(x9)) * x6) * small_t
            x3 = x3 + (math.sin(x8) * x4 - math.sin(x7) * math.cos(x8) * x5 - math.cos(x7) * math.cos(
                x8) * x6) * small_t
            x4 = x4 + (x12 * x5 - x11 * x6 - 9.81 * math.sin(x8)) * small_t
            x5 = x5 + (x10 * x6 - x12 * x4 + 9.81 * math.cos(x8) * math.sin(x7)) * small_t
            x6 = x6 + (x11 * x4 - x10 * x5 + 9.81 * math.cos(x8) * math.cos(x7) - 9.81 - u0 / 1.4) * small_t
            x7 = x7 + (x10 + (math.sin(x7) * (math.sin(x8) / math.cos(x8))) * x11 + (
                    math.cos(x7) * (math.sin(x8) / math.cos(x8))) * x12) * small_t
            x8 = x8 + (math.cos(x7) * x11 - math.sin(x7) * x12) * small_t
            x9 = x9 + ((math.sin(x7) / math.cos(x8)) * x11 + (math.cos(x7) / math.cos(x8)) * x12) * small_t
            x10 = x10 + (-0.92592592592593 * x11 * x12 + 18.51851851851852 * u1) * small_t
            x11 = x11 + (0.92592592592593 * x10 * x12 + 18.51851851851852 * u2) * small_t
            x12 = x12

            self.state = np.array(
                [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12],
                dtype=np.float32)
            # x1 = np.clip(x1, -2, 2)
            # x2 = np.clip(x2, -2, 2)
            # x3 = np.clip(x3, -2, 2)
            # x4 = np.clip(x4, -2, 2)
            # x5 = np.clip(x5, -2, 2)
            # x6 = np.clip(x6, -2, 2)
            # x7 = np.clip(x7, -2, 2)
            # x8 = np.clip(x8, -math.pi / 2 + 0.00001, math.pi / 2 - 0.00001)
            # x9 = np.clip(x9, -2, 2)
            # x10 = np.clip(x10, -2, 2)
            # x11 = np.clip(x11, -2, 2)
            # x12 = np.clip(x12, -2, 2)
            for item in self.state:
                if abs(item) > 20:
                    reward = -300
                    done = True
                    break

            time = round(time + small_t, 10)
            #

            if 1.06 + bias >= x3 >= 0.94 + bias:
                done = True
                break

        self.state = np.array(
            [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12],
            dtype=np.float32)

        # x3 [0.94, 1.06]

        reward += (5 - 2 * math.fabs(x3 - 1.05))

        # for item in self.state:
        #     if abs(item) >= 20:
        #         reward -= 1000
        #         break

        if 1.06 + bias >= x3 >= 0.94 + bias:
            reward = 1000
            done = True

        # if 1.1 >= x3 >= 0.9:
        #     reward = 1000
        #     done = True

        return self._get_obs(), reward, done, {}

    def reset(self):
        high = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0])
        # low = np.array([-0.4, -0.4, -0.4, -0.4, -0.4, -0.4, 0, 0, 0, 0, 0, 0])
        # low = np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0, 0, 0, 0, 0, 0])
        low = np.array([0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0, 0, 0, 0, 0, 0])
        self.state = self.np_random.uniform(low=low, high=high)

        return self._get_obs()

    def _get_obs(self):
        max_bound = 30
        min_bound = -max_bound
        # self.state[0] = np.clip(self.state[0], -5, 5)
        # self.state[1] = np.clip(self.state[1], -5, 5)
        # self.state[2] = np.clip(self.state[2], -5, 5)
        # self.state[3] = np.clip(self.state[3], -5, 5)
        # self.state[4] = np.clip(self.state[4], -5, 5)

        self.state[0] = np.clip(self.state[0], min_bound, max_bound)
        self.state[1] = np.clip(self.state[1], min_bound, max_bound)
        self.state[2] = np.clip(self.state[2], min_bound, max_bound)
        self.state[3] = np.clip(self.state[3], min_bound, max_bound)
        self.state[4] = np.clip(self.state[4], min_bound, max_bound)
        self.state[5] = np.clip(self.state[5], min_bound, max_bound)
        self.state[6] = np.clip(self.state[6], min_bound, max_bound)
        # cos做除数
        self.state[7] = np.clip(self.state[7], -1.55, 1.55)
        self.state[8] = np.clip(self.state[8], min_bound, max_bound)
        self.state[9] = np.clip(self.state[9], min_bound, max_bound)
        self.state[10] = np.clip(self.state[10], min_bound, max_bound)
        self.state[11] = np.clip(self.state[11], min_bound, max_bound)

        return self.state

    # def step_size(self, u, step_size=0.001):
    #
    #     done = False
    #
    #     offset = 0
    #     scala = 1
    #     u1 = u[0] - offset
    #     u1 = u1 * scala
    #
    #     t = 0.1
    #     time = 0
    #     state_list = []
    #     while time <= t:
    #         x1, x2, x3, x4 = self.state
    #         x1_new = x1 + x2 * step_size

    #         x3_new = x3 + x4 * step_size
    #         x4_new = x4 + u1 * step_size
    #         state_list.append([x1_new, x2_new])
    #         self.state = np.array([x1_new, x2_new, x3_new, x4_new], dtype=np.float32)
    #
    #         if 0.2 >= x1_new >= -0.1 and -0.6 >= x2_new >= -0.9:
    #             done = True
    #         time = round(time + step_size, 10)
    #     return self.state, state_list, done

# def angle_normalize(self, x):
#     return (( (x + np.pi) % (2 * np.pi) ) - np.pi)
