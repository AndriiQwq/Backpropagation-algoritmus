import numpy as np

def prepare_xor_data():
    """Initialize initial input and output for XOR problem"""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([[0], [1], [1], [0]])
    return X, Y

def get_xor_training_data():
    """Returns training data for XOR problem"""
    return [
        (np.array([0, 0]).reshape(1, -1), np.array([0]).reshape(1, -1)),
        (np.array([0, 1]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 0]).reshape(1, -1), np.array([1]).reshape(1, -1)),
        (np.array([1, 1]).reshape(1, -1), np.array([0]).reshape(1, -1))
    ]

def get_xor_test_data():
    """Returns test data for XOR problem"""
    return [
        (np.array([0, 0]).reshape(1, -1), 0),
        (np.array([0, 1]).reshape(1, -1), 1),
        (np.array([1, 0]).reshape(1, -1), 1),
        (np.array([1, 1]).reshape(1, -1), 0)
    ]

def initialize_weights_and_biases(layer_sizes):
    """
    layer_sizes: list of layer sizes, e.g. [2, 4, 1]
    Returns lists of weights and offsets for each layer.
    """
    weights = []
    biases = []
    for i in range(len(layer_sizes) - 1):
        W = np.random.randn(layer_sizes[i], layer_sizes[i+1])
        B = np.zeros((1, layer_sizes[i+1]))
        weights.append(W)
        biases.append(B)
    return weights, biases
