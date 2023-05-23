import numpy as np
import torch
import yaml

from network_train.acc6.acc_train import Agent as accAgent
from network_train.model2txt import yam2txt
from network_train.tora.tora_train import Agent as toraAgent
from network_train.b1.b1_train import Agent as b1Agent
from network_train.b2.b2_train import Agent as b2Agent
from network_train.b3.b3_train import Agent as b3Agent
from network_train.b4.b4_train import Agent as b4Agent
from network_train.b5.b5_train import Agent as b5Agent
from network_train.tora.tora_train import Agent as toraAgent
# from network_train.qmpc.qmpc_train import Agent as qmpcAgent
from network_train.quad12.quad12_train import Agent as quadAgent
from network_train.cartpole.cartpole_train import Agent as cartAgent

def model2yaml(model, output_name, tanh=True):
    # model.load_state_dict(torch.load(PATH))

    layer_count = 1
    dnn_dict = {}
    dnn_dict['weights'] = {}
    dnn_dict['offsets'] = {}
    dnn_dict['activations'] = {}
    layer_number = None
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
                dnn_dict['weights'][layer_number].append(a)
        # Bias
        else:
            dnn_dict['offsets'][layer_number] = []
            for row in numpy_para:
                dnn_dict['offsets'][layer_number].append(row.tolist())

        layer_count += 1
    if tanh:
        for i in range(layer_number):
            dnn_dict['activations'][i + 1] = 'Tanh'
        output_name = output_name + '_Tanh'
    else:
        for i in range(layer_number):
            dnn_dict['activations'][i + 1] = 'Relu'
        dnn_dict['activations'][layer_number] = 'Tanh'
        output_name = output_name + '_Relu'

    output_filename = 'polar_output/' + output_name + '.yml'

    with open(output_filename, 'w') as f:
        yaml.dump(dnn_dict, f)
    print(output_filename, 'finished')

    # with open('output/tanh20x20x20.yml', "r") as f1:
    #     content = f1.read()
    # yamlData = yaml.load(content,Loader = yaml.FullLoader)
    # with open(output_filename, 'w') as f:
    #     yaml.dump(yamlData,f)
    # print('finished')


if __name__ == '__main__':
    # agent = toraAgent()
    agent = b1Agent()
    agent = b2Agent()
    agent = b3Agent()
    agent = b4Agent()
    agent = b5Agent()
    # agent = toraAgent()
    # agent = qmpcAgent()
    agent = accAgent()
    agent = quadAgent()
    # agent = cartAgent()
    agent.load()
    small_size1 = '2_20'
    small_size2 = '3_20'
    small_size3 = '3_64'

    big_size1 = '2_100'
    big_size2 = '3_100'
    big_size3 = '3_200'
    big_size4 = '4_200'
    big_size5 = '4_100'


    tanh = True
    size = small_size3
    acti_f = '_Tanh.yml' if tanh else '_Relu.yml'
    verisig_name = 'quad_network7_' + size

    model2yaml(agent.actor, verisig_name, tanh=tanh)
    yam2txt(verisig_name + acti_f, 'nn_quad7_' + size, offset=0, scala=10)
