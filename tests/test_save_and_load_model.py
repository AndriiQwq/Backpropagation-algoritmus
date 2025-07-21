import sys
import os
import time
import numpy as np
from colorama import Fore, init
init(autoreset=True)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlp import MLP
from data_handler import prepare_xor_data, get_xor_training_data, get_xor_test_data, initialize_weights_and_biases
from config_manager import ConfigManager
from training import train_model
from model_manager import ModelManager

def test_model(model, test_data):
    correct = 0
    for input_set, label in test_data:
        model.forward(input_set)
        output = model.layers[-1]['a']
        if round(output.item()) == label:
            correct += 1
    return correct == len(test_data)

def main():
    print(Fore.CYAN + "Testing ModelManager save/load...\n")
    X, Y = prepare_xor_data()
    config = ConfigManager()
    manager = ModelManager(config)
    model = MLP(X, Y, config)

    # Init layers
    layer_sizes = config.layer_sizes
    activations = config.activations
    weights, biases = initialize_weights_and_biases(layer_sizes)
    for i in range(len(layer_sizes) - 1):
        model.create_layer(layer_sizes[i], layer_sizes[i+1], weights[i], biases[i], activation=activations[i])

    # Train and test
    train_model(model, config, get_xor_training_data())
    test_data = get_xor_test_data()
    assert test_model(model, test_data), Fore.RED + "Original model failed test!"

    # Save and load
    model_path = "test_xor_model_manager.npy"
    assert manager.save_model(model, model_path), Fore.RED + "ModelManager save failed!"
    loaded_model = MLP(X, Y, config)
    assert manager.load_model(loaded_model, model_path), Fore.RED + "ModelManager load failed!"
    assert test_model(loaded_model, test_data), Fore.RED + "Loaded model failed test!"

    # Compare outputs
    for input_set, _ in test_data:
        model.forward(input_set)
        loaded_model.forward(input_set)
        assert np.allclose(model.layers[-1]['a'], loaded_model.layers[-1]['a'], atol=1e-10), Fore.RED + "Outputs mismatch!"

    print(Fore.GREEN + "ModelManager save/load test PASSED!")

if __name__ == '__main__':
    start = time.time()
    main()
    print(Fore.MAGENTA + f'Total time: {time.time() - start:.2f} seconds')