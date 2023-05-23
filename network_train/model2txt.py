import numpy as np
import torch
import yaml

from network_train.tora.tora_train import Agent as toraAgent
from network_train.b1.b1_train import Agent as b1Agent
from network_train.b2.b2_train import Agent as b2Agent
from network_train.b3.b3_train import Agent as b3Agent
from network_train.b4.b4_train import Agent as b4Agent
from network_train.b5.b5_train import Agent as b5Agent
from network_train.tora.tora_train import Agent as toraAgent
# from network_train.qmpc.qmpc_train import Agent as qmpcAgent


# with open('output/tanh20x20x20.yml', "r") as f1:
#     content = f1.read()
# yamlData = yaml.load(content, Loader=yaml.FullLoader)
# print(yamlData)


def model2txt(model, output_name, input_dim=2, output_dim=1, num_hidden=2, hidden_neurons=20, offset=0, scala=1,
              tanh=True):
    # model.load_state_dict(torch.load(PATH))
    if tanh:
        output_name = output_name + '_tanh'
    else:
        output_name = output_name + '_relu'
    output_filename = 'output/' + output_name

    layer_count = 1
    dnn_dict = {}
    dnn_dict['weights'] = {}
    dnn_dict['offsets'] = {}
    dnn_dict['activations'] = {}
    layer_number = None
    with open(output_filename, 'w') as f:
        f.write(str(input_dim) + '\n')
        f.write(str(output_dim) + '\n')
        f.write(str(num_hidden) + '\n')
        for i in range(num_hidden):
            f.write(str(hidden_neurons) + '\n')
        for p in model.parameters():
            numpy_para = p.detach().cpu().numpy()
            # print(numpy_para)
            layer_number = int((layer_count + 1) / 2)
            # Weight
            if layer_count % 2 == 1:
                dnn_dict['weights'][layer_number] = []
                for row in numpy_para:
                    # print(type(row))
                    a = []
                    for col in row:
                        a.append(float(col))
                        f.write(str(float(col)) + '\n')
                    # dnn_dict['weights'][layer_number].append(a)
            # Bias
            else:
                dnn_dict['offsets'][layer_number] = []
                for row in numpy_para:
                    dnn_dict['offsets'][layer_number].append(row.tolist())
                    f.write(str(row.tolist()) + '\n')
            layer_count += 1
        f.write(str(offset) + '\n')
        f.write(str(scala) + '\n')

    # with open('output/tanh20x20x20.yml', "r") as f1:
    #     content = f1.read()
    # yamlData = yaml.load(content,Loader = yaml.FullLoader)
    # with open(output_filename, 'w') as f:
    #     yaml.dump(yamlData,f)
    # print('finished')


def yam2txt(input_name, output_name, offset, scala):
    with open('polar_output/' + input_name, "r") as f1:
        content = f1.read()
    yamlData = yaml.load(content, Loader=yaml.FullLoader)
    if yamlData['activations'][1] == 'Tanh':
        output_name = output_name + '_tanh'
    else:
        output_name = output_name + '_relu'
    output_filename = 'polar_output/' + output_name
    with open(output_filename, 'w') as f:
        offsets = yamlData['offsets']
        weights = yamlData['weights']
        f.write(str(len(weights[1][0])) + '\n')
        f.write(str(len(weights[len(weights)])) + '\n')
        f.write(str(len(weights) - 1) + '\n')

        for i in range(len(weights) - 1):
            f.write(str(len(weights[i + 1])) + '\n')

        activations = yamlData['activations']
        for i in activations:
            # f.write(activation_fun)
            if activations[i] == 'Tanh':
                f.write('tanh\n')
            else:
                f.write('ReLU\n')

        for i in range(len(weights)):
            for j in range(len(weights[i + 1])):
                for k in range(len(weights[i + 1][j])):
                    f.write(str(weights[i + 1][j][k]) + '\n')
                f.write(str(offsets[i + 1][j]) + '\n')
        f.write(str(offset) + '\n')
        f.write(str(scala) + '\n')
    print(output_filename, 'finished')


def yam2txt_reachnn(input_name, output_name, offset, scala):
    with open('output/' + input_name, "r") as f1:
        content = f1.read()
    yamlData = yaml.load(content, Loader=yaml.FullLoader)
    if yamlData['activations'][1] == 'Tanh':
        output_name = output_name + '_tanh'
    else:
        output_name = output_name + '_relu'
    output_filename = 'reachnn_output/' + output_name
    with open(output_filename, 'w') as f:
        offsets = yamlData['offsets']
        weights = yamlData['weights']
        f.write(str(len(weights[1][0])) + '\n')
        f.write(str(len(weights[len(weights)])) + '\n')
        f.write(str(len(weights) - 1) + '\n')

        for i in range(len(weights) - 1):
            f.write(str(len(weights[i + 1])) + '\n')

        for i in range(len(weights)):
            for j in range(len(weights[i + 1])):
                for k in range(len(weights[i + 1][j])):
                    f.write(str(weights[i + 1][j][k]) + '\n')
                f.write(str(offsets[i + 1][j]) + '\n')
        f.write(str(offset) + '\n')
        f.write(str(scala) + '\n')
    print(output_filename, 'finished')


if __name__ == '__main__':
    # agent = toraAgent()
    # agent = b1Agent()
    # agent = b2Agent()
    # agent = b3Agent()
    # agent = b4Agent()
    # agent = b5Agent()
    # agent = toraAgent()
    # agent = qmpcAgent()
    # agent.load()
    small_size1 = '2_20'
    small_size2 = '3_20'

    big_size1 = '2_100'
    big_size2 = '3_100'
    big_size3 = '3_200'
    big_size4 = '4_200'
    big_size5 = '4_100'
    # model2txt(agent.actor, 'nn_1_' + big_size2, input_dim=2, output_dim=1, num_hidden=3, hidden_neurons=100, offset=0,
    #           scala=4,
    #           tanh=True)
    # yam2txt('b1_network_3_100_Relu.yml', 'nn_acc_' + big_size2, offset=0, scala=2)
    yam2txt_reachnn('b1_network_3_100_Tanh.yml', 'reachnn_1_' + big_size2, offset=0, scala=2)
