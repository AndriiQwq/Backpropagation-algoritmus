import numpy as np

class Activation_functions:
    def __init__(self):
        """Determine the activation function"""
        self.activation_function = None

    def process_activation_function(self, Z, activation_function_name):
        """Function to processing choose of activation function"""
        if activation_function_name == 'Sigmoid':
            self.activation_function = self.Sigmoid(Z)
        elif activation_function_name == 'Tanh':
            self.activation_function = self.Tanh(Z)
        elif activation_function_name == 'ReLU':
            self.activation_function = self.ReLU(Z)

        return self.activation_function

    def process_derivation_of_activation_function(self, Z, activation_function_name):
        """Function to processing choose of derivation of activation function"""
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
        """https://www.delftstack.com/howto/python/relu-derivative-python/"""
        return np.where(Z > 0, 1, 0)

    def get_activation_function(self, layer):
        return self.process_activation_function(layer['Z'], layer['activation'])

    def get_derivation_of_activation_function(self, layer):
        return self.process_derivation_of_activation_function(layer['Z'], layer['activation'])

