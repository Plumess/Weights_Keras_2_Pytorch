import torch
import numpy as np
import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from model_pytorch_NIMA import NIMA


def print_keras_model(keras_model):
    for layer in keras_model.layers:
        print("layer.get_config():", layer.get_config()['name'])
        weights = layer.get_weights()
        if len(weights) == 2:
            print("layer.get_weights():", weights[0].shape, weights[1].shape)

def compare_weight(keras_model, pytorch_model, weight_name='conv2d_1.weight'):
    for name, param in pytorch_model.named_parameters():
        print(name)
        if name == weight_name:
            pyt_weight = param.detach().numpy()
            print("pyt_weight.shape:", pyt_weight.shape)
    for layer in keras_model.layers:
        if layer.get_config()['name'] == weight_name:
            if weight_name.split()[1] == 'weight':
                keras_weight = layer.get_weights()[0]
                keras_weight = np.transpose(keras_weight, (3, 2, 0, 1))
            elif weight_name.split()[1] == 'bias':
                keras_weight = layer.get_weights()[1]
    print("weight_dis", pyt_weight - keras_weight)

def keras_to_pyt(km, pm):
    weight_dict = dict()
    for layer in km.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.DepthwiseConv2D:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (2, 3, 0, 1))
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.BatchNormalization:
            if (len(layer.get_weights()) >= 1):
                weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
            if (len(layer.get_weights()) >= 2):
                weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            if (len(layer.get_weights()) >= 3):
                weight_dict[layer.get_config()['name'] + '.running_mean'] = layer.get_weights()[2]
            if (len(layer.get_weights()) >= 4):
                weight_dict[layer.get_config()['name'] + '.running_var'] = layer.get_weights()[3]
        elif type(layer) is keras.layers.ReLU:
            pass
        elif type(layer) is keras.layers.Dropout:
            pass

    pyt_state_dict = pm.state_dict()
    for key in pyt_state_dict.keys():
        print(key)
        if 'num_batches_tracked' in key:
            continue
        pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
    pm.load_state_dict(pyt_state_dict)
    return pm

def main():
    # define the model
    # keras
    image_size = 224
    base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
    for layer in base_model.layers:
        layer.trainable = False
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)
    keras_network = Model(base_model.input, x)
    keras_network.summary()
    keras_network.load_weights(r'.\mobilenet_weights.h5')
    print_keras_model(keras_network)

    # pytorch
    pytorch_network = NIMA()

    # transfer keras model to pytorch
    pytorch_network = keras_to_pyt(keras_network, pytorch_network)
    torch.save(pytorch_network.state_dict(), "NIMA_pytorch_model.pth")

main()