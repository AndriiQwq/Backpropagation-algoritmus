import numpy as np

def prepare_xor_data():
    """Initialize initial input and output for XOR problem"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    return X, Y

def initialize_weights_and_biases(input_size, hidden_layer_size):
    """Create matrix 2x4 for Weights and fill it with random or static values"""
    W1 = np.random.randn(input_size, hidden_layer_size) * np.sqrt(1 / input_size)
    W2 = np.random.randn(hidden_layer_size, 1) * np.sqrt(1 / hidden_layer_size)

    B1 = np.zeros((1, hidden_layer_size))
    B2 = np.zeros((1, 1))
    
    return W1, W2, B1, B2
