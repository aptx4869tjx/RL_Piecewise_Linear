import json
from itertools import chain

import torch

# from abstract.b1.b1_abs import Agent as b1Agent
from conversion.hybridautomata import HybridAT
from conversion.hybridjump import HybridJump
from conversion.hybridmode import HybridMode
from verify.divide_tool import initiate_divide_tool, str_to_list


def read_json_config(file_path):
    with open(file_path, 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
        # print(fcc_data['dynamics'])
        # print(fcc_data)
        return fcc_data


def generate_mode(sys_data, action_dim=1):
    mode = HybridMode(sys_data['name'])
    mode.dynamics = ''
    for index, d in enumerate(sys_data['dynamics']):
        mode.dynamics += sys_data['state_vars'][index] + '\' = ' + d + '\n'
    if action_dim == 1:
        mode.dynamics += 'a\' = 0.0\n' + 'clock\' = 1.0'
    elif action_dim > 1:
        for i in range(action_dim):
            mode.dynamics += 'a' + str(i + 1) + '\' = 0.0\n'
        mode.dynamics+='clock\' = 1.0'
    mode.invariant = 'clock <= ' + str(sys_data['control_step'])
    mode.generate_mode_str()
    return mode


def extract_policy(divide_tool, network):
    return 0


# def generate_jumps(state_space, initial_intervals, agent, sys_data):
#     divide_tool = initiate_divide_tool(state_space, initial_intervals)
#
#     abstract_states = divide_tool.intersection(list(chain(*state_space)))
#     jump_list = []
#     for abstract_state in abstract_states:
#         abs_list = str_to_list(abstract_state)
#         dim = len(abs_list)
#         half_dim = int(dim / 2)
#         abs_tensor = torch.tensor(abs_list, dtype=torch.float).unsqueeze(0)
#         # [0.9959851 - 0.22767536 - 1.4451199]
#         # tensor([0.9960, -0.2277, -1.4451])
#         coefficient = agent.actor.cal_coefficients(abs_tensor).squeeze(0).detach().numpy()
#         jump = HybridJump(sys_data)
#         guard_str = 'clock = ' + str(sys_data['control_step']) + '  '
#         for i in range(half_dim):
#             guard_str += str(abs_list[i]) + ' - ' + sys_data['state_vars'][i] + ' <= 0  '  # lower bound
#             guard_str += sys_data['state_vars'][i] + ' <= ' + str(abs_list[i + half_dim]) + '  '  # upper bound
#         jump.guard = guard_str
#         reset_str = 'a\' := ' + str(coefficient[0])  # bias
#         for i in range(half_dim):
#             reset_str += ' + ' + str(coefficient[i + 1]) + ' * ' + sys_data['state_vars'][i]
#         reset_str += '\n' + 'clock\' := 0.0'
#         jump.reset = reset_str
#         jump.generate_jump_str()
#         jump_list.append(jump)
#
#     return jump_list


def generate_jumps(state_space, initial_intervals, agent, sys_data, action_dim=1):
    divide_tool = initiate_divide_tool(state_space, initial_intervals)

    abstract_states = divide_tool.intersection(list(chain(*state_space)))
    jump_list = []
    for abstract_state in abstract_states:
        abs_list = str_to_list(abstract_state)
        dim = len(abs_list)
        half_dim = int(dim / 2)
        abs_tensor = torch.tensor(abs_list, dtype=torch.float).unsqueeze(0)
        # [0.9959851 - 0.22767536 - 1.4451199]
        # tensor([0.9960, -0.2277, -1.4451])
        coefficient = agent.actor.cal_coefficients(abs_tensor).squeeze(0).detach().numpy()
        jump = HybridJump(sys_data)
        guard_str = 'clock = ' + str(sys_data['control_step']) + '  '
        for i in range(half_dim):
            guard_str += str(abs_list[i]) + ' - ' + sys_data['state_vars'][i] + ' <= 0  '  # lower bound
            guard_str += sys_data['state_vars'][i] + ' <= ' + str(abs_list[i + half_dim]) + '  '  # upper bound
        jump.guard = guard_str
        if action_dim == 1:
            reset_str = 'a\' := ' + str(coefficient[0])  # bias
            for i in range(half_dim):
                reset_str += ' + ' + str(coefficient[i + 1]) + ' * ' + sys_data['state_vars'][i]
        elif action_dim > 1:
            print(action_dim)
            assert (action_dim * (half_dim + 1) == coefficient.size)
            reset_str = ''
            for j in range(action_dim):
                start_dim = j * (half_dim + 1)
                reset_str += 'a' + str(j + 1) + '\' := ' + str(coefficient[start_dim])
                for i in range(half_dim):
                    reset_str += ' + ' + str(coefficient[start_dim + i + 1]) + ' * ' + sys_data['state_vars'][i]
                reset_str += '\n'
        else:
            exit("action dimension is not valid")

        reset_str += '\n' + 'clock\' := 0.0'
        jump.reset = reset_str
        jump.generate_jump_str()
        jump_list.append(jump)

    return jump_list


# single mode
def generate_hybrid_automata(mode, jump_list, sys_data, action_dim):
    hybrid = HybridAT(sys_data, action_dim)
    hybrid.modes = [mode]
    hybrid.jumps = jump_list
    # hybrid.output_modes()
    # hybrid.output_jumps()
    # hybrid.output_setting_str()
    # hybrid.output_init_str()
    # hybrid.output_hybrid_str()
    return hybrid


def convert2hybrid(config_file, state_space, initial_intervals, agent, action_dim=1):
    sys_data = read_json_config(config_file)
    mode = generate_mode(sys_data,action_dim)
    jump_list = generate_jumps(state_space, initial_intervals, agent, sys_data, action_dim)
    hybrid_sys = generate_hybrid_automata(mode, jump_list, sys_data, action_dim)
    return hybrid_sys


if __name__ == '__main__':
    # state_space = [[-2.6, -2.5], [2.4, 2.5]]
    # initial_intervals = [2.5, 2.5]
    # agent = b1Agent()
    # agent.load()
    # hybrid = convert2hybrid('b1.json', state_space, initial_intervals, agent)
    # hybrid.output_hybrid_str()
    print("finished")
