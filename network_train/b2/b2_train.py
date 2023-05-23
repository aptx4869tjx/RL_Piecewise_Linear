import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from abstract_env.b2 import B2Env
from network_train.noisy.evaluate import reward_evaluate, evaluate_with_noisy, evaluate_multiple

hiden_size = 100
hidden_layer = 3
hiden_size = 20
hidden_layer = 2
script_path = os.path.split(os.path.realpath(__file__))[0]
relu = False
if relu:
    pt_file0 = os.path.join(script_path, "b2-relu-actor" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path, "b2-relu-critic" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path, "b2-relu-actor-target" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path, "b2-relu-critic-target" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
else:
    pt_file0 = os.path.join(script_path, "b2-actor" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path, "b2-critic" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path, "b2-actor-target" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path, "b2-critic-target" + str(hidden_layer) + "_" + str(hiden_size) + ".pt")

env = B2Env()
env.reset()


# env.render()

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        large_weight = 0.02
        if hidden_layer == 2:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear1.weight.data.normal_(0, large_weight)
            self.linear1.bias.data.zero_()
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear2.weight.data.normal_(0, large_weight)
            self.linear2.bias.data.zero_()
            self.linear3 = nn.Linear(hidden_size, output_size)
            self.linear3.weight.data.normal_(0, large_weight)
            self.linear3.bias.data.zero_()
        else:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear1.weight.data.normal_(0, large_weight)
            self.linear1.bias.data.zero_()
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear2.weight.data.normal_(0, large_weight)
            self.linear2.bias.data.zero_()
            self.linear3 = nn.Linear(hidden_size, hidden_size)
            self.linear3.weight.data.normal_(0, large_weight)
            self.linear3.bias.data.zero_()
            self.linear4 = nn.Linear(hidden_size, output_size)
            self.linear4.weight.data.normal_(0, large_weight)
            self.linear4.bias.data.zero_()

    def forward(self, s):
        if relu:
            if hidden_layer == 2:
                x = torch.relu(self.linear1(s))
                x = torch.relu(self.linear2(x))
                x = torch.tanh(self.linear3(x))
            else:
                x = torch.relu(self.linear1(s))
                x = torch.relu(self.linear2(x))
                x = torch.relu(self.linear3(x))
                x = torch.tanh(self.linear4(x))
        else:
            if hidden_layer == 2:
                x = torch.tanh(self.linear1(s))
                x = torch.tanh(self.linear2(x))
                x = torch.tanh(self.linear3(x))
            else:
                x = torch.tanh(self.linear1(s))
                x = torch.tanh(self.linear2(x))
                x = torch.tanh(self.linear3(x))
                x = torch.tanh(self.linear4(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class Agent(object):
    def __init__(self):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.noisy = [0, 0]

    def update_egreed(self):
        self.e_greed = max(0.01, self.e_greed - 0.002)

    def reset(self):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.noisy = [0, 0]

    def save(self):  # 保存网络的参数数据
        torch.save(self.actor.state_dict(), pt_file0)
        torch.save(self.critic.state_dict(), pt_file1)
        torch.save(self.actor_target.state_dict(), pt_file2)
        torch.save(self.critic_target.state_dict(), pt_file3)
        # print(pt_file + " saved.")

    def load(self):  # 加载网络的参数数据
        self.actor.load_state_dict(torch.load(pt_file0))
        self.network.load_state_dict(torch.load(pt_file0))
        self.critic.load_state_dict(torch.load(pt_file1))
        self.actor_target.load_state_dict(torch.load(pt_file2))
        self.critic_target.load_state_dict(torch.load(pt_file3))
        print(pt_file3 + " loaded.")

    def act(self, s0):
        # s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        return a0

    def put(self, *transition):
        if len(self.buffer) == self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return

        samples = random.sample(self.buffer, self.batch_size)

        s0, a0, r1, s1 = zip(*samples)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        s1 = torch.tensor(s1, dtype=torch.float)

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + self.gamma * self.critic_target(s1, a1).detach()

            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_pred, y_true)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)


def evaluate(agent, episode=100):
    min_reward = 0
    crash = False
    reward_list = []
    success = 0
    for l in range(episode):
        reward = 0
        s0 = env.reset()
        reach = False
        for step in range(100):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            reward += r1
            s0 = s1
            if done:
                print(l, ' : reach goal', s1, reward)
                reach = True
                success += 1
                break

        reward_list.append(reward)
        if reward <= -600:
            crash = True
        if not reach:
            print('Not reach goal!!!--------------------------------', reward)
    print(success, '/', 1000)
    print('crash: ', crash)
    print('avg reward: ', np.mean(reward_list))
    return np.min(reward_list), crash


def train_model(agent, max_epi=600, terminate_pre=True):
    reward_list = []
    for j in range(30):
        agent.reset()
        for episode in range(max_epi):
            agent.update_egreed()
            s0 = env.reset()
            episode_reward = 0
            step_size = 0
            for step in range(30):
                # env.render()
                if np.random.rand() < agent.e_greed:
                    a0 = [(np.random.rand() - 0.5) * 2]
                else:
                    a0 = agent.act(s0)
                # a0 = agent.act(s0)
                s1, r1, done, _ = env.step(a0)
                step_size += 1

                agent.put(s0, a0, r1, s1)

                episode_reward += r1
                s0 = s1
                if step % 4 == 0:
                    agent.learn()
                if done:
                    break
            if episode % 5 == 4:
                agent.save()
            reward_list.append(episode_reward)
            print(episode, ': ', episode_reward, step_size)
            if terminate_pre and episode >= 10 and np.min(reward_list[-8:]) >= 94:
                min_reward, _ = evaluate(agent)
                if min_reward > 90:
                    agent.save()
                    return
        if not terminate_pre:
            return reward_list

            # divide_tool = initiate_divide_tool(state_space, initial_intervals)


def evaluate_trace(agent, episode=100):
    trace_list = []
    success = 0
    for l in range(episode):
        s0 = env.reset()
        reach = False
        states_one_ep = []
        for step in range(10000):
            # env.render()
            a0 = agent.act(s0)
            s1, states, done = env.step_size(a0)
            states_one_ep += states
            if done:
                success += 1
                reach = True
                print(l, ':', step, s1)
                break
            s0 = s1
        if not reach:
            print('Not reach goal!!!--------------------------------')
        trace_list.append(states_one_ep)
    print(success, '/', 1000)
    return np.array(trace_list)


if __name__ == "__main__":
    unsafe = False
    agent = Agent()
    #
    # t0 = time.time()
    # train_model(agent)
    # t1 = time.time()
    # print(t1 - t0)
    # agent.load()
    # evaluate(agent)
    # trace_list = evaluate_trace(agent, episode=100)
    # np.save('./b2_Tanh_3_100_trace_list', arr=trace_list)

    start_time = time.time()
    repeat_list = reward_evaluate(agent, train_model, max_epi=400)
    np.save('./b2_train_reward', arr=repeat_list)
    end_time = time.time()
    print(end_time - start_time)  # 33

    # noise_arr = [0.0002, 0.0002]
    # state_space = [[-2.5, -2.5], [2.5, 2.5]]
    # multiple_train_list = evaluate_multiple(agent, train_model, evaluate, env, state_space, noise_arr, step=20)
    # # np.save('./b1_multiple_train_noisy_list' + str(noise_arr), arr=multiple_train_list)
    # np.save('./b2_noisy_list' + str(noise_arr), arr=multiple_train_list)

    print('finished')
