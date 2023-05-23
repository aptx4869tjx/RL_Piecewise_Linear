import os

import gym
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from abstract_env.ACC import ACCEnv
from conversion.convert import convert2hybrid
from verify.divide_tool import initiate_divide_tool, str_to_list

hiden_size = 100
hidden_layer = 4
hiden_size = 20
hidden_layer = 3
relu = False

state_space = [[-300, -100, -100, 0, -50, -50], [300, 100, 100, 300, 100, 100]]
initial_intervals = [1, 0.1, 0.1, 1, 0.1, 0.1]
# initial_intervals = [2, 0.2, 0.2, 2, 0.2, 0.2]
# initial_intervals = [3, 0.3, 0.3, 3, 0.3, 0.3]
# initial_intervals = [4, 0.4, 0.4, 4, 0.4, 0.4]
# initial_intervals = [0.5, 0.05, 0.05, 0.5, 0.05, 0.05]
# initial_intervals = [0.2, 0.02, 0.02, 0.2, 0.02, 0.02]
# initial_intervals = [0.7, 0.07, 0.07, 0.7, 0.07, 0.07]
initial_intervals = [600, 200, 200, 300, 100, 100]
script_path = os.path.split(os.path.realpath(__file__))[0]
if relu:
    pt_file0 = os.path.join(script_path,
                            "acc6_relu_abs-actor1" + "_" + str(initial_intervals) + "_" + str(hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path, "acc6_relu_abs-critic1" + "_" + str(initial_intervals) + "_" + str(
        hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path,
                            "acc6_relu_abs-actor-target1" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path,
                            "acc6_relu_abs-critic-target1" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(hiden_size) + ".pt")
else:
    pt_file0 = os.path.join(script_path,
                            "acc6_abs-actor1" + "_" + str(initial_intervals) + "_" + str(hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path,
                            "acc6_abs-critic1" + "_" + str(initial_intervals) + "_" + str(hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path, "acc6_abs-actor-target1" + "_" + str(initial_intervals) + "_" + str(
        hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path, "acc6_abs-critic-target1" + "_" + str(initial_intervals) + "_" + str(
        hidden_layer) + "_" + str(hiden_size) + ".pt")

# initial_intervals = [0.2, 0.01, 0.01, 0.2, 0.01, 0.01]
env = ACCEnv()
env.reset()


# env.render()

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        large_weight = 0.1
        self.input_size = input_size
        self.output_size = output_size
        as_dim = int(input_size / 2)
        if hidden_layer == 3:
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.linear1.weight.data.normal_(0, large_weight)
            self.linear1.bias.data.zero_()
            self.linear2 = nn.Linear(hidden_size, hidden_size)
            self.linear2.weight.data.normal_(0, large_weight)
            self.linear2.bias.data.zero_()
            self.linear3 = nn.Linear(hidden_size, hidden_size)
            self.linear3.weight.data.normal_(0, large_weight)
            self.linear3.bias.data.zero_()
            self.linear4 = nn.Linear(hidden_size, as_dim + 1)
            self.linear4.weight.data.normal_(0, large_weight)
            self.linear4.bias.data.zero_()
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
            self.linear4 = nn.Linear(hidden_size, hidden_size)
            self.linear4.weight.data.normal_(0, large_weight)
            self.linear4.bias.data.zero_()
            self.linear5 = nn.Linear(hidden_size, as_dim + 1)
            self.linear5.weight.data.normal_(0, large_weight)
            self.linear5.bias.data.zero_()

    def forward(self, s):
        action, _ = self.fw_imp(s)
        t = _.tolist()
        return action

    def fw_imp(self, s):
        tmp = s[:, self.input_size:]
        x = self.cal_coefficients(s[:, 0:self.input_size])
        ones = torch.ones((tmp.size(0), 1))
        cat = torch.cat([ones, tmp], dim=-1)
        y = cat.mul(x)
        res = torch.sum(y, dim=1, keepdim=True)
        return res, x

    def cal_coefficients(self, s):
        if relu:
            if hidden_layer == 3:
                x = torch.relu(self.linear1(s))
                x = torch.relu(self.linear2(x))
                x = torch.relu(self.linear3(x))
                x = torch.tanh(self.linear4(x))
            else:
                x = torch.relu(self.linear1(s))
                x = torch.relu(self.linear2(x))
                x = torch.relu(self.linear3(x))
                x = torch.relu(self.linear4(x))
                x = torch.tanh(self.linear5(x))

        else:
            if hidden_layer == 3:
                x = torch.tanh(self.linear1(s))
                x = torch.tanh(self.linear2(x))
                x = torch.tanh(self.linear3(x))
                x = torch.tanh(self.linear4(x))
            else:
                x = torch.tanh(self.linear1(s))
                x = torch.tanh(self.linear2(x))
                x = torch.tanh(self.linear3(x))
                x = torch.tanh(self.linear4(x))
                x = torch.tanh(self.linear5(x))
        b = torch.Tensor([1.0, 1, 1, 1, 1.0, 1.0, 1.0])
        x = x * b
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
    def __init__(self, divide_tool):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32

        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]
        dim = self.env.observation_space.shape[0]

        self.divide_tool = divide_tool

        self.actor = Actor(s_dim, hiden_size, a_dim)
        self.network = Actor(s_dim, hiden_size, a_dim)
        self.actor_target = Actor(s_dim, hiden_size, a_dim)
        self.critic = Critic(s_dim + dim + a_dim, hiden_size, a_dim)
        self.critic_target = Critic(s_dim + dim + a_dim, hiden_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.noisy = [0, 0]

    def reset(self):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.capacity = 10000
        self.batch_size = 32

        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]
        dim = self.env.observation_space.shape[0]
        self.actor = Actor(s_dim, hiden_size, a_dim)
        self.network = Actor(s_dim, hiden_size, a_dim)
        self.actor_target = Actor(s_dim, hiden_size, a_dim)
        self.critic = Critic(s_dim + dim + a_dim, hiden_size, a_dim)
        self.critic_target = Critic(s_dim + dim + a_dim, hiden_size, a_dim)
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
        abs = str_to_list(self.divide_tool.get_abstract_state(s0))
        abs_s0 = abs + s0.tolist()
        # s0 = torch.tensor(s0, dtype=torch.float).unsqueeze(0)
        s0 = torch.tensor(abs_s0, dtype=torch.float).unsqueeze(0)
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


# 29.896785949285455 29.9474327496227
def evaluate(agent, epi=100):
    min_reward = 0
    reward_list = []
    success = 0
    for l in range(epi):
        reward = 0
        s0 = env.reset()
        reach = False
        for step in range(60):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            # print(s1)
            if done:
                success += 1
                print('reach goal', s1, step)
                reach = True
                break
            reward += r1
            s0 = s1
        if not reach:
            print('Not reach goal!!!--------------------------------')
        reward_list.append(reward)
    print('avg reward: ', np.mean(reward_list), success, '/', epi)
    return np.min(reward_list)


def evaluate_trace(agent, epi=100):
    min_reward = 0
    reward_list = []
    trace_list = []
    success = 0
    for l in range(epi):
        reward = 0
        trace = []
        s0 = env.reset()
        trace.append(s0)
        reach = False
        for step in range(60):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            trace.append(s1)
            # print(s1)
            if done:
                success += 1
                print('reach goal', s1, step)
                reach = True
                break
            reward += r1
            s0 = s1
        if not reach:
            print('Not reach goal!!!--------------------------------')
        reward_list.append(reward)
        trace_list.append(trace)
    print('avg reward: ', np.mean(reward_list), success, '/', epi)
    return np.array(trace_list)


def train_model(agent, max_epi=10000, terminate_pre=True):
    reward_list = []
    for j in range(30):
        agent.reset()
        for episode in range(max_epi):
            s0 = env.reset()
            episode_reward = 0
            ab_s = agent.divide_tool.get_abstract_state(s0)
            step_size = 0
            for step in range(50):
                # env.render()
                a0 = agent.act(s0)
                s1, r1, done, _ = env.step(a0)
                # print(s1)
                step_size += 1
                next_abs = agent.divide_tool.get_abstract_state(s1)
                agent.put(str_to_list(ab_s) + s0.tolist(), a0, r1, str_to_list(next_abs) + s1.tolist())
                episode_reward += r1
                s0 = s1
                ab_s = next_abs
                if step % 8 == 0:
                    agent.learn()
                if done:
                    break
            if episode % 8 == 7:
                agent.save()
            reward_list.append(episode_reward)
            print(episode, ': ', episode_reward, step_size)
            if terminate_pre and episode >= 3 and np.min(reward_list[-3:]) >= 0:
                min_reward = evaluate(agent)
                if min_reward > 0:
                    agent.save()
                    return [], []
                break

            # divide_tool = initiate_divide_tool(state_space, initial_intervals)


if __name__ == "__main__":
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    # agent.load()
    train_model(agent)
    agent.load()
    evaluate(agent)
    hybrid = convert2hybrid('acc.json', state_space, initial_intervals, agent)
    hybrid.output_hybrid_str()
