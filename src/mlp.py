import numpy as np
from activation_functions import Activation_functions
from utils.logger import get_logger
from model_manager import ModelManager 

class MLP:
    def __init__(self, X, Y, config=None):
        self.X = X
        self.Y = Y
        self.layers = []

        self.config = config
        self.logger = get_logger("MLP", config) if config else None
        self.manager = ModelManager(config)

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
        """Maybe was better create recursive function for hidden layers, but before create function for last layer.
            with returned values of error for the next layer. In this stap we need to calculate three gradients
            relatively to error function MSE: (for weights, bias and predicted output
            and recalculate the weights and biases for each layer)

            @error - is a error for last(output) layer"""

        """For output layer"""
        activation_function_method = Activation_functions()
        derivation_of_activation_function = activation_function_method.get_derivation_of_activation_function(
            self.layers[-1])

        delta_output_error = error * derivation_of_activation_function

        """Gradient off loss function to weight, (last layer input) * (delta==error * (f'(Z)))"""
        grad_MSE_W = np.dot(self.layers[-2]['a'].T, delta_output_error)
        """For bigger batch size use sum for each row, keepdims by default is False, 
        source: https://dnmtechs.com/understanding-the-keepdims-parameter-in-numpy-sum/"""
        grad_MSE_B = np.sum(delta_output_error, axis=0, keepdims=True)
        #grad_MSE_a = np.dot(delta_output_error, self.layers[-1]['W'].T)

        """Updating weights for output layer"""
        current_layer = self.layers[-1]
        self.update_weights(current_layer, grad_MSE_W, grad_MSE_B)

        """For next layer"""
        """So, for documentation, we need tests OR, AND, XOR problems with different layers,
         to do this create loop to recalculate weights and biases for each layer, 
         maybe it wasn't better way to do this, but it's simple """
        iterator = len(self.layers)
        for _ in range(len(self.layers) - 1):
            """ layers[iterator - 3] - Layer before current layer
                layers[iterator - 2] - Current layer (if iterator = len(self.layers),
                                       then current layer is layer after output layer)
                layers[iterator - 1] - Next layer"""
            current_layer = self.layers[iterator - 2]

            derivation_of_activation_function_hidden = activation_function_method.get_derivation_of_activation_function(
                current_layer)

            error_for_hidden_layer = np.dot(delta_output_error, self.layers[iterator-1]['W'].T)

            """@next_delta_hidden_layer - dMSE/dah * f'(Zh)"""
            next_delta_hidden_layer = error_for_hidden_layer * derivation_of_activation_function_hidden

            """Control if we reach first(input) layer and don't have any more layers"""
            if iterator == 2:
                """First layer, don't have previous layer"""
                grad_MSE_W = np.dot(self.X.T, next_delta_hidden_layer)
            else:
                grad_MSE_W = np.dot(self.layers[iterator - 3]['a'].T, next_delta_hidden_layer)
            grad_MSE_B = np.sum(next_delta_hidden_layer, axis=0, keepdims=True)

            """Updating weights for hidden layer"""
            self.update_weights(current_layer, grad_MSE_W, grad_MSE_B)

            """Updating values of MSE_a gradient to distribute it for next layer"""
            delta_output_error = next_delta_hidden_layer
            """Update iterator, go back to the next hidden layer to previous and update values of waiting and biases"""
            iterator -= 1

    def update_weights(self, current_layer, grad_MSE_W, grad_MSE_B):
        if self.config.use_momentum:
            """Momentum use a previous values of gradients with %"""
            current_layer['vW'] = self.config.momentum * current_layer['vW'] - self.config.learning_rate * grad_MSE_W
            current_layer['vB'] = self.config.momentum * current_layer['vB'] - self.config.learning_rate * grad_MSE_B
            current_layer['W'] += current_layer['vW']
            current_layer['B'] += current_layer['vB']
        else:
            current_layer['W'] -= self.config.learning_rate * grad_MSE_W
            current_layer['B'] -= self.config.learning_rate * grad_MSE_B

    def MSE_Loss_evaluating(self):
        """MSE loss function"""
        predicated_output = self.layers[-1]['a']
        """Where len(self.Y) is a batch size, Y is correct output, predicated_output is last output(y) - predication"""
        error = np.power(self.Y - predicated_output, 2).sum() / (len(self.Y) * 2)
        """So, here we have function with elements: (d - y)^2, is equal to (y - d)^2!!!
            , where d is correct output, y is predicated output.
        """
        return error

    def get_last_layer_error(self):
        predicted_output = self.layers[-1]['a']
        Y_label = self.Y
        last_error = predicted_output - Y_label
        return last_error
    
    def save_model(self, file_path):
        """Save model to file using ModelManager"""
        return self.manager.save_model(self, file_path)

    def load_model(self, file_path):
        """Load model from file using ModelManager"""
        return self.manager.load_model(self, file_path)

    def get_model_info(self, file_path=None):
        """Get information about current model or a saved model"""
        if file_path is None:
            # Show info about current model
            return self.manager.get_current_model_info(self)
        else:
            # Show info about saved model
            return self.manager.get_saved_model_info(file_path)