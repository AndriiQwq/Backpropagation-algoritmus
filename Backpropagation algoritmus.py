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
show_confusion_matrix = False
learning_rate = 0
momentum = 0
activation_function_name = None

default_config = {
    'Settings': {
        'epoch_count': '20',
        'show_confusion_matrix': 'False',
        'learning_rate': '0.01',
        'momentum': '0.9',
        'activation_function_name': 'Sigmoid'
    }
}


def get_config():
    global epoch_count, show_confusion_matrix, learning_rate, momentum, activation_function_name

    epoch_count = config.getint('Settings', 'epoch_count')
    show_confusion_matrix = config.getboolean('Settings', 'show_confusion_matrix')
    learning_rate = config.getfloat('Settings', 'learning_rate')
    momentum = config.getfloat('Settings', 'momentum')
    activation_function_name = config.get('Settings', 'activation_function_name')


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

    def process_activation_function(self, Z):
        if activation_function_name == 'Sigmoid':
            self.activation_function = self.Sigmoid(Z)
        elif activation_function_name == 'Tanh':
            self.activation_function = self.Tanh(Z)
        elif activation_function_name == 'ReLU':
            self.activation_function = self.ReLU(Z)

        return self.activation_function

    def Sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def Tanh(self, Z):
        return np.exp(Z) - np.exp(-Z) / np.exp(Z) + np.exp(-Z)

    def ReLU(self, Z):
        return np.maximum(0, Z)

    def get_activation_function(self, Z):
        return self.process_activation_function(Z)


class MLP:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.layers = []

    def create_layer(self, input_size, output_size, W, B):
        layer = {
            'input_size': input_size,
            'output_size': output_size,
            'W': W,
            'B': B,
            'Z': None,
            'h': None
        }

        self.layers.append(layer)
        return layer

    def show_layers_information(self):
        for layer in self.layers:
            print(Fore.LIGHTYELLOW_EX + f'Layer {self.layers.index(layer) + 1}')
            print(Fore.LIGHTGREEN_EX +
                  f"Size: {layer['input_size']}x{layer['output_size']},\n Weight: {layer['W']},\n Bias: {layer['B']}\n")

    def forward(self, x):
        """Forward input x to each layer, each layer recalculate inputs values and forward to the up layer"""
        for layer in self.layers:
            """Calculate intermediate result"""
            layer['Z'] = np.dot(x, layer['W']) + layer['B']
            """x = layer['Z']"""

            """Z: [z1, z2, z3, z4]"""

            """Forward result to activation function(Sigmoid, Tanh, ReLU)"""
            activation_function_method = Activation_functions()
            layer['h'] = activation_function_method.get_activation_function(layer['Z'])

            """h: [a1, a2, a3, a4]"""
            """Will be put how input to the next layer"""
            x = layer['h']




    def backward(self, x):
        pass

    def MSE_Loss(self):
        pass


"""Loss function"""


def plot_metrics(train_losses):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()


def train_model(model, epochs=10):
    training_losses = []
    evaluation_accuracies = []

    for epoch in range(epochs):
        model.train()

        """Training information"""
        training_info = {
            'running_loss': 0.0,
            'correct': 0,
            'total': 0
        }

        for inputs, labels in train_loader:
            #optimizer.zero_grad()

            outputs = model(inputs)
            loss = MSE_LOSS(outputs, labels)
            #loss.backward()
            #optimizer.step()

            """Append training information"""
            training_info['running_loss'] += loss.item()
            # predicated output
            training_info['total'] += labels.size(0)
            training_info['correct'] += (predicted == labels).sum().item()

        training_accuracy = training_info['correct'] / training_info['total']
        training_losses.append(training_info['running_loss'] / len(train_loader))

        """Evaluation model"""
        evaluation_accuracy = evaluate_model(model)
        evaluation_accuracies.append(evaluation_accuracy)

        """Logging"""
        print(Fore.GREEN + "--------------------------------------------------------------------------------\n",
              Fore.LIGHTMAGENTA_EX + f"Epoch {epoch + 1},",
              Fore.LIGHTRED_EX + f" Train Loss: {training_info['running_loss'] / len(train_loader):.4f}, ",
              Fore.LIGHTGREEN_EX + f"Train Accuracy: {training_accuracy:.4f}, ",
              Fore.LIGHTCYAN_EX + f"Test Accuracy: {evaluation_accuracy:.4f}",
              Fore.GREEN + "\n--------------------------------------------------------------------------------")

    plot_metrics(training_losses)
    return model


def evaluate_model(model):
    model.eval()

    training_info = {
        'correct': 0,
        'total': 0
    }

    evaluation_accuracy = training_info['correct'] / training_info['total']
    return evaluation_accuracy


if __name__ == '__main__':
    start_time = time.time()

    """Initialize initial input and output for XOR problem"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])

    """Initialize model"""
    model = MLP(X, Y)

    """Create matrix 2x4 for Weights and fill it with random values from -0.1 to 0.1"""
    W1 = np.random.uniform(-0.1, 0.1, (2, 4))
    W2 = np.random.uniform(-0.1, 0.1, (4, 1))

    B1 = np.array([0, 0, 0, 0])
    B2 = np.array([0])
    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    L1 = model.create_layer(2, 4, W1, B1)
    L2 = model.create_layer(4, 1, W2, B2)

    model.show_layers_information()
    model.forward(X)
    model.show_layers_information()


    #train_model(model, epochs=epoch_count)
    #for epoch in range(epoch_count):  # epoch_count???

    # while True:
    #     evaluate_loss_function_L(model)
    #     compute_derivation_W_b_x_for_each_layer(model)
    #     Adjust_parametrs_W_b_using_learning_rate()
    #     if all_inputs_used:
    #         if stopping_condition_met:
    #             break

    end_time = time.time()

    print(Fore.MAGENTA + f'Training time: {end_time - start_time:.2f} seconds')
