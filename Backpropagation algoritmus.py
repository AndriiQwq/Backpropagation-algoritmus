import numpy as np
import matplotlib.pyplot as plt

import time
import os
import configparser
from colorama import Fore, init

init(autoreset=True)

config_file = 'config.ini'
config = configparser.ConfigParser()

"""Parameters for the configuration file"""
epoch_count = 0
learning_rate = 0
momentum = 0
first_activation_function_name = 'Sigmoid'
second_activation_function_name = 'Sigmoid'
use_momentum = False

default_config = {
    'Settings': {
        'epoch_count': '500',
        'show_confusion_matrix': 'False',
        'learning_rate': '0.01',
        'momentum': '0.9',
        'use_momentum': 'True',
        'first_activation_function_name(Sigmoid, Tanh, ReLU)': 'Sigmoid',
        'second_activation_function_name(Sigmoid, Tanh, ReLU)': 'Sigmoid',
    }
}


def get_config():
    global epoch_count, show_confusion_matrix, learning_rate, momentum, first_activation_function_name, use_momentum, second_activation_function_name

    epoch_count = config.getint('Settings', 'epoch_count')
    learning_rate = config.getfloat('Settings', 'learning_rate')
    momentum = config.getfloat('Settings', 'momentum')
    use_momentum = config.getboolean('Settings', 'use_momentum')
    first_activation_function_name = config.get('Settings', 'first_activation_function_name(Sigmoid, Tanh, ReLU)')
    second_activation_function_name = config.get('Settings', 'second_activation_function_name(Sigmoid, Tanh, ReLU)')


if not os.path.exists(config_file):
    config.read_dict(default_config)
    with open(config_file, 'w') as configfile:
        config.write(configfile)
    print(Fore.LIGHTYELLOW_EX + 'Default configuration was created\n')
    get_config()
else:
    print(Fore.LIGHTYELLOW_EX + 'Configuration configured from config file\n')
    config.read(config_file)
    get_config()


class Activation_functions:
    def __init__(self):
        """Determine the activation function"""
        self.activation_function = None

    def process_activation_function(self, Z, activation_function_name):
        if activation_function_name == 'Sigmoid':
            self.activation_function = self.Sigmoid(Z)
        elif activation_function_name == 'Tanh':
            self.activation_function = self.Tanh(Z)
        elif activation_function_name == 'ReLU':
            self.activation_function = self.ReLU(Z)

        return self.activation_function

    def process_derivation_of_activation_function(self, Z, activation_function_name):
        if activation_function_name == 'Sigmoid':
            self.activation_function = self.Sigmoid_derivation(Z)
        elif activation_function_name == 'Tanh':
            self.activation_function = self.Tanh_derivation(Z)
        elif activation_function_name == 'ReLU':
            self.activation_function = self.ReLU_derivation(Z)

        return self.activation_function

    def Sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def Tanh(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def Sigmoid_derivation(self, Z):
        Sigmoid_value = self.Sigmoid(Z)
        return Sigmoid_value * (1 - Sigmoid_value)

    def Tanh_derivation(self, Z):
        Tanh_value = self.Tanh(Z)
        return 1 - np.power(Tanh_value, 2)

    def ReLU_derivation(self, Z):
        return np.where(Z > 0, 1, 0)

    def get_activation_function(self, layer):
        return self.process_activation_function(layer['Z'], layer['activation'])

    def get_derivation_of_activation_function(self, layer):
        return self.process_derivation_of_activation_function(layer['Z'], layer['activation'])


class MLP:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.layers = []
        self.v = []

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def create_layer(self, input_size, output_size, W, B, activation):
        """Where first two values is a matrix size, W is weight, B is bias, Z is intermediate result, a is output"""
        layer = {
            'input_size': input_size,
            'output_size': output_size,
            'W': W,
            'B': B,
            'Z': None,
            'a': None,
            'error': None,
            'vW': np.zeros_like(W),
            'vB': np.zeros_like(B),
            'activation': activation
        }

        self.layers.append(layer)
        return layer

    def show_layers_information(self):
        for layer in self.layers:
            print(Fore.LIGHTYELLOW_EX + f'Layer {self.layers.index(layer) + 1}')
            print(Fore.LIGHTGREEN_EX +
                  f"Size: {layer['input_size']}x{layer['output_size']},"
                  f"\n Weight: {layer['W']},\n Bias: {layer['B']},\n"
                  f" Z: {layer['Z']},\n a: {layer['a']}\n")

    def forward(self, x):
        """Forward input x to each layer, each layer recalculate inputs values and forward to the up layer"""

        """For first layer"""
        """Input x is 4x2 matrix(if input bs 4):
            [[0, 0], 
             [0, 1], 
             [1, 0], 
             [1, 1]]
        """
        """Input x is 1x2 matrix:
            [[0, 0], 
        """
        """Bias is 1x4 matrix: [0, 0, 0, 0]"""
        """W is 2x4 matrix with random numbers"""
        """mxn * nxp = mxp"""
        for layer in self.layers:
            """Calculate intermediate result, np.dot for multiplication matrix"""
            layer['Z'] = np.dot(x, layer['W']) + layer['B']
            """x = layer['Z']"""

            """For first layer"""
            """Z is 4x4 matrix, after multiplication and adding: 
                [z11, z12, z13, z14
                 z21, z22, z23, z24
                 z31, z32, z33, z34
                 z41, z42, z43, z44]
            """
            """Z is 1x4 matrix: 
                [z11, z12, z13, z14]
            """
            """Forward result to activation function(Sigmoid, Tanh, ReLU)"""
            activation_function_method = Activation_functions()
            layer['a'] = activation_function_method.get_activation_function(layer)

            """For first layer"""
            """h is 4x4 matrix: [a11, a12, a13, a14
                   a21, a22, a23, a24
                   a31, a32, a33, a34
                   a41, a42, a43, a44]"""
            """h is 1x4 matrix: [a11, a12, a13, a14]"""

            """Will be put how input to the next layer"""
            x = layer['a']

            """For last layer, the value a is output y(labels)"""
            """x -> model -> y"""

        return x

    def backward(self, error):
        #error = self.layers[-1]['a'] - self.Y

        """For output layer"""
        activation_function_method = Activation_functions()
        derivation_of_activation_function = activation_function_method.get_derivation_of_activation_function(
            self.layers[-1])

        delta_output_error = error * derivation_of_activation_function

        """Gradient off loss function to weight, (last layer input) * (delta==error * (f'(Z)))"""
        grad_MSE_W = np.dot(self.layers[-2]['a'].T, delta_output_error)
        """For bigger batch size"""
        grad_MSE_B = np.sum(delta_output_error, axis=0, keepdims=True)
        #grad_MSE_a = delta_output_error * self.layers[-1]['W']

        """Updating weights for output layer"""
        current_layer = self.layers[-1]
        self.update_weights(current_layer, grad_MSE_W, grad_MSE_B)

        """For next layer"""
        derivation_of_activation_function_hidden = activation_function_method.get_derivation_of_activation_function(
            self.layers[-2])

        error_for_hidden_layer = np.dot(delta_output_error, self.layers[-1]['W'].T)
        next_delta_hidden_layer = error_for_hidden_layer * derivation_of_activation_function_hidden

        grad_MSE_W = np.dot(self.X.T, next_delta_hidden_layer)
        grad_MSE_B = np.sum(delta_output_error, axis=0, keepdims=True)
        # grad_MSE_a = delta_output_error * self.layers[-1]['W']

        """Updating weights for hidden layer"""
        current_layer = self.layers[-2]
        self.update_weights(current_layer, grad_MSE_W, grad_MSE_B)

    def update_weights(self, current_layer, grad_MSE_W, grad_MSE_B):
        if use_momentum:
            current_layer['vW'] = momentum * current_layer['vW'] - learning_rate * grad_MSE_W
            current_layer['vB'] = momentum * current_layer['vB'] - learning_rate * grad_MSE_B
            current_layer['W'] += current_layer['vW']
            current_layer['B'] += current_layer['vB']
        else:
            current_layer['W'] -= learning_rate * grad_MSE_W
            current_layer['B'] -= learning_rate * grad_MSE_B

    def MSE_Loss_evaluating(self):
        """MSE loss function"""
        predicated_output = self.layers[-1]['a']
        """Where len(self.Y) is a batch size, Y is correct output, predicated_output is last output(y) - predication"""
        error = np.power(self.Y - predicated_output, 2).sum() / (len(self.Y) * 2)
        """So, here we have function with elements: (d - y)^2, is equal to (y - d)^2!!!
            , where d is correct output, y is predicated output.
        """
        return error

    def test_model(self, test_data):
        correct = 0
        print(Fore.GREEN + "-------------------------------------------------")
        for input_set, label in test_data:
            # model.set_X(input_set)
            # model.set_Y(label)

            model.forward(input_set)
            """a is 1x4 matrix: [a11, a12, a13, a14]"""
            output = self.layers[-1]['a']

            if round(output.item()) == label:
                print(Fore.BLUE + f"Input set: {input_set}, Label: {label}",
                      Fore.GREEN + f"Correct prediction: {output} == {label}")
                correct += 1
            else:
                print(Fore.BLUE + f"Input set: {input_set}, Label: {label}",
                      Fore.RED + f"Incorrect prediction: {output} != {label}")
        print(Fore.GREEN + "-------------------------------------------------")

        if correct == len(test_data):
            print(Fore.GREEN + "Correct model")
        else:
            print(Fore.RED + "Incorrect model")

    def get_last_layer_error(self):
        predicted_output = model.layers[-1]['a']
        Y_label = self.Y
        last_error = predicted_output - Y_label
        return last_error


if __name__ == '__main__':
    start_time = time.time()

    """Initialize initial input and output for XOR problem"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    """Initialize model"""
    model = MLP(X, Y)

    hidden_layer_size = 4

    """Create matrix 2x4 for Weights and fill it with random values"""
    W1 = np.random.randn(2, hidden_layer_size) * np.sqrt(1 / 2)
    W2 = np.random.randn(hidden_layer_size, 1) * np.sqrt(1 / 4)

    B1 = np.zeros((1, hidden_layer_size))
    B2 = np.zeros((1, 1))
    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    L1 = model.create_layer(2, hidden_layer_size, W1, B1, activation=first_activation_function_name)
    L2 = model.create_layer(hidden_layer_size, 1, W2, B2, activation=second_activation_function_name)

    """Training data"""
    training_data = [
        (np.array([0, 0]).reshape(1, -1), np.array([0]).reshape(1, -1)),
        (np.array([0, 1]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 0]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 1]).reshape(1, -1), np.array([0]).reshape(1, -1))
    ]
    training_data_size = len(training_data)
    losses = []

    for epoch in range(epoch_count):
        """Reshuffle the training data"""
        np.random.shuffle(training_data)

        total_error = 0

        for i in range(training_data_size):
            #for input_set, label in training_data:
            model.set_X(training_data[i][0])
            model.set_Y(training_data[i][1])

            model.forward(training_data[i][0])

            error = model.get_last_layer_error()
            model.backward(error)

            total_error += model.MSE_Loss_evaluating()


        average = total_error / len(training_data)
        losses.append(total_error / len(training_data))

        """Check stopping condition"""
        if average < 0.00001 or epoch == 20000:
            print("We reached the optimal model")
            break

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch + 1}/{epoch_count}, Loss: {average:.5f}")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Progress')
    plt.show()

    """Testing model"""
    model.test_model(training_data)

    end_time = time.time()
    print(Fore.MAGENTA + f'Training time: {end_time - start_time:.2f} seconds')
