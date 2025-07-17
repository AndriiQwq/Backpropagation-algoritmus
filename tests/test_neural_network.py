import sys
import os
import time
import random
from colorama import Fore, init
init(autoreset=True)

# Adding src path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# And add imports 
from mlp import MLP
from data_utils import prepare_xor_data, initialize_weights_and_biases
from config_manager import ConfigManager
from training import train_model

def test_model(model, test_data):
    correct = 0
    print(Fore.GREEN + "-------------------------------------------------")
    for input_set, label in test_data:
        model.forward(input_set)
        """a is 1x4 matrix: [a11, a12, a13, a14]"""
        output = model.layers[-1]['a']

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

def show_layers_information(model):
    for layer in model.layers:
        print(Fore.LIGHTYELLOW_EX + f'Layer {model.layers.index(layer) + 1}')
        print(Fore.LIGHTGREEN_EX +
                f"Size: {layer['input_size']}x{layer['output_size']},"
                f"\n Weight: {layer['W']},\n Bias: {layer['B']},\n"
                f" Z: {layer['Z']},\n a: {layer['a']}\n")

if __name__ == '__main__':
    """Testing model"""
    start_time = time.time()

    """Prepare XOR data"""
    X, Y = prepare_xor_data()
    config = ConfigManager()

    """Initialize model"""
    model = MLP(X, Y, config)
    
    hidden_layer_size = 4
    W1, W2, B1, B2 = initialize_weights_and_biases(2, hidden_layer_size)
    
    model.create_layer(2, hidden_layer_size, W1, B1, activation=config.first_activation_function_name)
    model.create_layer(hidden_layer_size, 1, W2, B2, activation=config.second_activation_function_name)

    """Train model"""
    print(Fore.CYAN + "Training model...")
    losses = train_model(model, config)

    """Prepare test data"""
    test_data = [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0)
    ]
    random.shuffle(test_data)

    """Test model"""
    print(Fore.CYAN + "\nTesting model...")
    test_model(model, test_data)

    """Show layers information"""
    print(Fore.CYAN + "\nLayers information:")
    show_layers_information(model)

    end_time = time.time()
    print(Fore.MAGENTA + f'Total time: {end_time - start_time:.2f} seconds')