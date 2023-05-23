import os
import sys
import time

from abstract.noisy.evaluate import evaluate_multiple, reward_evaluate
from conversion.convert import convert2hybrid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 获取文件所在的当前路径
from abstract_env.cartpole_continuous import CartPoleContinuousEnv
from verify.divide_tool import initiate_divide_tool, str_to_list, initiate_divide_tool_rtree

hiden_size = 200
hidden_layer = 4
hiden_size = 64
hidden_layer = 3

state_space = [[-5, -10, -0.5, -10], [5, 10, 0.5, 10]]
# state_space = [[-6, -10, -0.6, -10], [6, 10, 0.6, 10]]

# initial_intervals = [1.5, 1.5, 0.15, 1.5]
# initial_intervals = [2.5, 2.5, 0.25, 2.5]
# initial_intervals = [5, 5, 0.5, 5]
initial_intervals = [5, 10, 0.5, 10]
# initial_intervals = [2.5, 10, 0.25, 10]
# initial_intervals = [5, 20, 0.5, 10]
# initial_intervals = [0.5, 1, 0.05, 1]
# initial_intervals = [1, 2, 0.1, 2]
# initial_intervals = [0.25, 0.5, 0.025, 0.5]
print(initial_intervals)

relu = False
script_path = os.path.split(os.path.realpath(__file__))[0]
if relu:
    pt_file0 = os.path.join(script_path,
                            "cartpole7_relu_abs-actor" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path,
                            "cartpole7_relu_abs-critic" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path, "cartpole7_relu_abs-actor-target" + "_" + str(initial_intervals) + "_" + str(
        hidden_layer) + "_" + str(hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path,
                            "cartpole7_relu_abs-critic-target" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(hiden_size) + ".pt")
else:
    pt_file0 = os.path.join(script_path,
                            "cartpole7_abs-actor" + "_" + str(initial_intervals) + "_" + str(hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file1 = os.path.join(script_path,
                            "cartpole7_abs-critic" + "_" + str(initial_intervals) + "_" + str(hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file2 = os.path.join(script_path,
                            "cartpole7_abs-actor-target" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(
                                hiden_size) + ".pt")
    pt_file3 = os.path.join(script_path,
                            "cartpole7_abs-critic-target" + "_" + str(initial_intervals) + "_" + str(
                                hidden_layer) + "_" + str(
                                hiden_size) + ".pt")

env = CartPoleContinuousEnv()
env.reset()


# env.render()

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Actor, self).__init__()
        large_weight = 1
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
        x = self.cal_coefficients(s[:, 0:self.input_size])

        tmp = s[:, self.input_size:]
        ones = torch.ones((tmp.size(0), 1))
        cat = torch.cat([ones, tmp], dim=-1)
        y = cat.mul(x)
        res = torch.sum(y, dim=1, keepdim=True)
        # res =torch.tanh(res)
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
        b = torch.Tensor([1.0, 2.0, 1.0, 10.0, 1.0])
        x = x * b
        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class Agent(object):
    def __init__(self, divide_tool):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.05
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5
        dim = self.env.observation_space.shape[0]
        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]

        self.divide_tool = divide_tool
        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def update_egreed(self):
        self.e_greed = max(0.0001, self.e_greed - 0.001)

    def reset(self):
        self.env = env
        self.gamma = 0.99
        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.05
        self.capacity = 10000
        self.batch_size = 32
        self.e_greed = 0.5

        s_dim = self.env.observation_space.shape[0] * 2
        a_dim = self.env.action_space.shape[0]
        dim = self.env.observation_space.shape[0]
        hide_size = hiden_size
        self.actor = Actor(s_dim, hide_size, a_dim)
        self.network = Actor(s_dim, hide_size, a_dim)
        self.actor_target = Actor(s_dim, hide_size, a_dim)
        self.critic = Critic(s_dim + dim + a_dim, hide_size, a_dim)
        self.critic_target = Critic(s_dim + dim + a_dim, hide_size, a_dim)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.buffer = []

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

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


def evaluate(agent):
    min_reward = 0
    fail = 0
    min_x1 = 100
    max_x1 = -100
    min_x3 = 100
    max_x3 = -100
    for l in range(100):
        reward = 0
        s0 = env.reset()
        safe = True

        for step in range(1000):
            # env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)
            if s1[0] < min_x1:
                min_x1 = s1[0]
            if s1[0] > max_x1:
                max_x1 = s1[0]
            if s1[2] < min_x3:
                min_x3 = s1[2]
            if s1[2] > max_x3:
                max_x3 = s1[2]
            if done:
                print(l, ':unsafe', s1, step)
                fail += 1
                safe = False
                break
            reward += r1
            s0 = s1
        if safe:
            print(l, ':safe', s0, min_x1, max_x1, min_x3, max_x3)
    print(fail, '/', 100)
    return fail == 0


def evaluate_trace(agent):
    min_reward = 0
    trace_list = []
    for l in range(100):
        reward = 0

        states_one_ep = []
        s0 = env.reset()
        states_one_ep = [[0, s0[2]]]
        safe = True
        for step in range(50):
            # env.render()
            a0 = agent.act(s0)
            small_t = 0.00005
            s1, states, done = env.step_size(a0, step_size=small_t)
            for j in range(len(states)):
                states[j][0] = round(step * 0.02 + (j + 1) * small_t, 10)
            states_one_ep += states
            if done:
                print('unsafe', s1, step)
                safe = False
                break
            s0 = s1

        if safe:
            print(l, '：safe')
        trace_list.append(states_one_ep)
    return np.array(trace_list)


def train_model(agent, max_epi=15000, terminate_pre=True):
    reward_list = []
    for j in range(100):
        agent.reset()
        for episode in range(max_epi):
            agent.update_egreed()
            s0 = env.reset()
            episode_reward = 0
            ab_s = agent.divide_tool.get_abstract_state(s0)
            step_size = 0
            for step in range(200):
                if np.random.rand() < agent.e_greed:
                    a0 = [(np.random.rand() - 0.5) * 2]
                else:
                    a0 = agent.act(s0)
                # a0 = agent.act(s0)
                s1, r1, done, _ = env.step(a0)
                step_size += 1
                next_abs = agent.divide_tool.get_abstract_state(s1)
                agent.put(str_to_list(ab_s) + s0.tolist(), a0, r1, str_to_list(next_abs) + s1.tolist())
                episode_reward += r1
                s0 = s1
                ab_s = next_abs
                if step % 16 == 0:
                    agent.learn()
                if done:
                    break
            if episode % 16 == 0:
                agent.save()
            reward_list.append(episode_reward)
            print(episode, ': ', int(episode_reward), step_size)
            if terminate_pre and episode >= 500 and np.min(reward_list[-5:]) >= 200:
                res = evaluate(agent)
                return
                if res:
                    agent.save()
                    return
                else:
                    break
        if not terminate_pre:
            return reward_list

            # divide_tool = initiate_divide_tool(state_space, initial_intervals)


if __name__ == "__main__":
    divide_tool = initiate_divide_tool(state_space, initial_intervals)
    agent = Agent(divide_tool)
    # agent.load()
    train_model(agent)
    agent.load()

    # evaluate(agent)
    trace_list = evaluate_trace(agent)
    np.save('./cart_trace_list7', arr=trace_list)
    t0=time.time()
    hybrid = convert2hybrid('cartpole.json', state_space, initial_intervals, agent)
    hybrid.output_hybrid_str()
    t1 = time.time()
    print(t1 - t0)

    # noise_arr = [0.0001, 0.0001, 0.0001, 0.0001]
    # multiple_train_list = evaluate_multiple(agent, train_model, evaluate, env, state_space, noise_arr)
    # np.save('./cart_pwl_' + str(initial_intervals) + 'multiple_train_noisy_list' + str(noise_arr),
    #         arr=multiple_train_list)

    # start_time = time.time()
    # repeat_list = reward_evaluate(agent, train_model, max_epi=3000) #247 815.3
    # np.save('./cart_train_reward_' + str(initial_intervals), arr=repeat_list)
    # end_time = time.time()
    # print(end_time - start_time)
    print('finished')
