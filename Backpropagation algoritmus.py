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
        'epoch_count': '500',
        'show_confusion_matrix': 'False',
        'learning_rate': '0.01',
        'momentum': '0.9',
        'activation_function_name(Sigmoid, Tanh, ReLU)': 'Sigmoid'
    }
}


def get_config():
    global epoch_count, show_confusion_matrix, learning_rate, momentum, activation_function_name

    epoch_count = config.getint('Settings', 'epoch_count')
    show_confusion_matrix = config.getboolean('Settings', 'show_confusion_matrix')
    learning_rate = config.getfloat('Settings', 'learning_rate')
    momentum = config.getfloat('Settings', 'momentum')
    activation_function_name = config.get('Settings', 'activation_function_name(Sigmoid, Tanh, ReLU)')


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

    def process_derivation_of_activation_function(self, Z):
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
        # if Z > 0:
        #     return 1
        # else:
        #     return 0
        return np.where(Z > 0, 1, 0)

    def get_activation_function(self, Z):
        return self.process_activation_function(Z)

    def get_derivation_of_activation_function(self, Z):
        return self.process_derivation_of_activation_function(Z)


class MLP:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.layers = []

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def create_layer(self, input_size, output_size, W, B):
        """Where first two values is a matrix size, W is weight, B is bias, Z is intermediate result, a is output"""
        layer = {
            'input_size': input_size,
            'output_size': output_size,
            'W': W,
            'B': B,
            'Z': None,
            'a': None,
            'error': None
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
            layer['a'] = activation_function_method.get_activation_function(layer['Z'])

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

    def backward(self):
        predicated_output = self.layers[-1]['a']

        activation_function_method = Activation_functions()
        derivation_of_activation_function = activation_function_method.get_derivation_of_activation_function(
            self.layers[-1]['Z'])

        grad_MSE_0y = 2 * (predicated_output - self.Y) / len(self.Y)

        b = grad_MSE_0y * derivation_of_activation_function

        grad_MSE_W = b * (self.layers[-2]['a']).T  # grad_z_w

        grad_MSE_B = b

        grad_MSE_a = b * self.layers[-1]['W']

        """Updating weights for output layer"""
        self.layers[-1]['W'] -= learning_rate * grad_MSE_W
        self.layers[-1]['B'] -= learning_rate * grad_MSE_B

        it = len(self.layers)  # last layer has index 2
        for i in range(it - 1):
            it -= 1

            derivation_of_activation_function = activation_function_method.get_derivation_of_activation_function(
                self.layers[-it - 1]['Z'])

            grad_MSE_0y = 2 * (self.layers[-it - 1]['a'] - self.Y) / len(self.Y)
            b = grad_MSE_0y * derivation_of_activation_function

            grad_MSE_B = b

            grad_MSE_a = b * self.layers[-it - 1]['W']

            if it == 1:
                grad_MSE_W = b * (self.X).T
            else:
                grad_MSE_W = b * (self.layers[-it - 2]['a']).T  # grad_z_w

            """Updating weights for output layer"""
            self.layers[-it - 1]['W'] -= learning_rate * grad_MSE_W
            self.layers[-it - 1]['B'] -= learning_rate * grad_MSE_B

        # output_delta = output_error * derivation_of_activation_function
        #
        # for i in range(len(self.layers) - 1, -1, -1):
        #     current_layer = self.layers[i]
        #
        #     if i == 0:
        #         layer_input = self.X
        #     else:
        #         layer_input = self.layers[i - 1]['a']
        #
        #     grad_W = np.dot(layer_input.T, output_delta)
        #     grad_B = np.sum(output_delta, axis=0, keepdims=True)
        #
        #     current_layer['W'] -= learning_rate * grad_W
        #     current_layer['B'] -= learning_rate * grad_B
        #
        #     if i > 0:
        #         output_delta = np.dot(output_delta, current_layer['W'].T) * \
        #                        activation_function_method.get_derivation_of_activation_function(self.layers[i - 1]['Z'])









        # predicated_output = self.layers[-1]['a']
        #
        # activation_function_method = Activation_functions()
        # derivation_of_activation_function = activation_function_method.get_derivation_of_activation_function(self.layers[-1]['Z'])
        #
        # output_error = 2*(predicated_output - self.Y) / len(self.Y)
        #
        # #error_output = (predicated_output - self.Y) * derivation_of_activation_function
        #
        # # grad_W = np.dot(self.layers[-2]['a'].T, error_output)
        # # grad_B = np.sum(error_output, axis=0, keepdims=True)
        #
        # # self.layers[-1]['W'] -= learning_rate * grad_W
        # # self.layers[-1]['B'] -= learning_rate * grad_B
        # output_delta = output_error * derivation_of_activation_function
        #
        # for i in range(len(self.layers)-1, -1, -1):
        #     current_layer = self.layers[i]
        #
        #     if i == 0:
        #         layer_input = self.X
        #     else:
        #         layer_input = self.layers[i-1]['a']
        #
        #     grad_W = np.dot(layer_input.T, output_delta)
        #     grad_B = np.sum(output_delta, axis=0, keepdims=True)
        #
        #     current_layer['W'] -= learning_rate * grad_W
        #     current_layer['B'] -= learning_rate * grad_B
        #
        #     if i > 0:
        #         output_delta = np.dot(output_delta, current_layer['W'].T) * \
        #                     activation_function_method.get_derivation_of_activation_function(self.layers[i-1]['Z'])



    def MSE_Loss_evaluating(self):
        """MSE loss function"""
        predicated_output = self.layers[-1]['a']
        """Where len(self.Y) is a batch size, Y is correct output, predicated_output is last output(y) - predication"""
        error = np.power(self.Y - predicated_output, 2).sum() / len(self.Y)
        """So, here we have function with elements: (d - y)^2, is equal to (y - d)^2!!!
            , where d is correct output, y is predicated output.
        """
        return error


    def test_model(self, test_data):
        self.forward(test_data)
        output = model.layers[-1]['a']
        print(Fore.GREEN + f'Test MSE: {self.MSE_Loss_evaluating()}')
        return output

    def test_xor_network(model):
        # Test cases for XOR
        test_cases = [
            ([0, 0], 0),
            ([0, 1], 1),
            ([1, 0], 1),
            ([1, 1], 0)
        ]

        print(Fore.CYAN + "\nXOR Gate Testing Results:")
        print(Fore.CYAN + "=" * 50)
        print(Fore.YELLOW + f"{'Input':^15} | {'Expected':^10} | {'Predicted':^10} | {'Correct':^10}")
        print(Fore.CYAN + "-" * 50)

        correct = 0
        for inputs, expected in test_cases:
            # Reshape input for network
            test_input = np.array(inputs).reshape(1, -1)

            # Forward pass
            model.forward(test_input)
            predicted = model.layers[-1]['a'][0][0]

            # Round prediction for binary classification
            predicted_binary = round(predicted)
            is_correct = predicted_binary == expected

            if is_correct:
                correct += 1
                result_color = Fore.GREEN
            else:
                result_color = Fore.RED

            print(result_color + f"{str(inputs):^15} | {expected:^10.4f} | {predicted:^10.4f} | {str(is_correct):^10}")

        accuracy = correct / len(test_cases) * 100
        print(Fore.CYAN + "-" * 50)
        print(Fore.GREEN + f"Overall Accuracy: {accuracy:.2f}%")
        print(Fore.CYAN + "=" * 50)



def plot_metrics():
    pass



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

    B1 = np.zeros((1, 4))
    B2 = np.zeros((1, 1))
    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    L1 = model.create_layer(2, 4, W1, B1)
    L2 = model.create_layer(4, 1, W2, B2)

    """Training model"""
    training_data = [
        (np.array([0, 0]).reshape(1, -1), np.array([0]).reshape(1, -1)),
        (np.array([0, 1]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 0]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 1]).reshape(1, -1), np.array([0]).reshape(1, -1))
    ]
    losses = []

    for epoch in range(epoch_count):
        """Reshuffle the training data"""
        np.random.shuffle(training_data)

        total_error = 0
        for i in range(1):
            for input_set, label in training_data:
                model.set_X(input_set)
                model.set_Y(label)

                model.forward(input_set)
                model.backward()

                total_error += model.MSE_Loss_evaluating()

        average = total_error / len(training_data)
        losses.append(total_error / len(training_data))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epoch_count}, Loss: {average:.6f}")

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Progress')
    plt.show()

    #
    # #model.show_layers_information()
    #
    # model.set_X(X[0])
    # model.set_Y(Y[0])
    # model.forward(X[0])

    # error_rate = model.MSE_Loss_evaluating()
    # print(Fore.GREEN + f'Error rate: {error_rate}')
    # #model.show_layers_information()
    #
    # model.backward()
    #
    # model.show_layers_information()



    model.test_xor_network()



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
