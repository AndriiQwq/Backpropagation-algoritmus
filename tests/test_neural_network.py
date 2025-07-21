import sys
import os
import time
from colorama import Fore, init
init(autoreset=True)

import numpy as np

# Adding src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from mlp import MLP
from data_handler import prepare_xor_data, get_xor_training_data, get_xor_test_data, initialize_weights_and_biases
from config_manager import ConfigManager
from training import train_model
from utils.testing import test_model

def show_layers_information(model):
    for idx, layer in enumerate(model.layers):
        print(Fore.LIGHTYELLOW_EX + f'Layer {idx + 1}')
        print(Fore.LIGHTGREEN_EX +
              f"Size: {layer['input_size']}x{layer['output_size']},"
              f"\n Weight: {layer['W']},\n Bias: {layer['B']},\n"
              f" Z: {layer['Z']},\n a: {layer['a']}\n")

if __name__ == '__main__':
    start_time = time.time()

    # Prepare XOR data
    X, Y = prepare_xor_data()
    config = ConfigManager()

    # Initialize model
    model = MLP(X, Y, config)

    # Get layer sizes and activations from config
    layer_sizes = config.layer_sizes
    activations = config.activations

    # Initialize weights and biases for any number of layers
    weights, biases = initialize_weights_and_biases(layer_sizes)

    # Dynamically create layers
    for i in range(len(layer_sizes) - 1):
        model.create_layer(
            layer_sizes[i],
            layer_sizes[i+1],
            weights[i],
            biases[i],
            activation=activations[i]
        )

    # Get training data
    training_data = get_xor_training_data()

    # Train model
    print(Fore.CYAN + "Training model...")
    losses = train_model(model, config, training_data)

    # Get test data (теперь из data_handler)
    test_data = get_xor_test_data()

    # Test model
    print(Fore.CYAN + "\nTesting model...")
    test_model(model, test_data)

    # Show layers information
    print(Fore.CYAN + "\nLayers information:")
    show_layers_information(model)
    
    model.get_model_info()

    end_time = time.time()
    print(Fore.MAGENTA + f'Total time: {end_time - start_time:.2f} seconds')