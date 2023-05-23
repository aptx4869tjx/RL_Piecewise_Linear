import random
import time

import numpy as np


def evaluate_with_noisy(self, env, state_space, noise_step_arr, noise_num=100, episode=20, step=500):
    iteration_mean_reward_list = []
    noise = [0 for i in range(len(state_space[0]))]
    for j in range(noise_num):
        reward_list = []
        print('evaluate iteration: ', j, 'noise level', noise)
        for i in range(episode):
            total_reward = 0
            obs = env.reset()
            obs = add_noisy(obs, noise)
            obs = clip(obs, state_space)
            step_size = 0
            for t in range(step):
                act = self.act(obs)
                obs, reward, done, _ = env.step(act)
                obs = add_noisy(obs, noise)
                obs = clip(obs, state_space)
                total_reward += reward
                step_size += 1
                if done:
                    break
            reward_list.append(total_reward)
            # print('EVALUATE: ', i, step_size)
        print('AVG REWARD: ', np.mean(reward_list))
        iteration_mean_reward_list.append(np.array(reward_list))
        noise = update_noisy(noise, noise_step_arr)
    return np.array(iteration_mean_reward_list)


def evaluate_multiple(self, train_model, evaluate, env, state_space, noise_step_arr, noise_num=100, episode=20,
                      step=500):
    multiple_train_list = []
    for i in range(10):
        print("train iteration:", i + 1, "start")
        self.reset()
        train_model(self)
        self.load()
        evaluate(self)
        noisy_list = evaluate_with_noisy(self, env, state_space, noise_step_arr, noise_num, episode, step)
        multiple_train_list.append(noisy_list)
    return multiple_train_list


def clip(state, state_space1):
    for i in range(len(state)):
        state[i] = np.clip(state[i], state_space1[0][i], state_space1[1][i])
    return state


def add_noisy(obs, noise):
    for i in range(len(obs)):
        obs[i] += random.gauss(0, noise[i])
    return obs


def update_noisy(noise, noise_step_arr):
    # noisy = [0.005, 0.01, 0.0005, 0.01]
    # step_length = [0.005, 0.005, 0.005, 0.005]
    for i in range(len(noise)):
        noise[i] += noise_step_arr[i]
    return noise


def reward_evaluate(agent, train_model, max_epi, repeat=5):
    repeat_list = []
    t0 = time.time()
    for i in range(repeat):
        print('Training Reward Evaluation: ', i + 1)
        start_time = time.time()
        reward_list = train_model(agent, max_epi, False)
        repeat_list.append(reward_list)
        end_time = time.time()
        print(end_time - start_time)
    t1 = time.time()
    print('overall time:', t1 - t0)
    return np.array(repeat_list)
