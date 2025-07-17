from mlp import MLP
from data_utils import prepare_xor_data, initialize_weights_and_biases
from config_manager import ConfigManager
from training import train_model
from visualizer import vizualize_training_process

def main():
    """Prepare XOR data"""
    X, Y = prepare_xor_data()

    """Initialize config"""
    config = ConfigManager()

    """Initialize model"""
    model = MLP(X, Y, config)

    hidden_layer_size = 4
    W1, W2, B1, B2 = initialize_weights_and_biases(2, hidden_layer_size)

    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    model.create_layer(2, hidden_layer_size, W1, B1, activation=config.first_activation_function_name)
    model.create_layer(hidden_layer_size, 1, W2, B2, activation=config.second_activation_function_name)

    """Train model"""
    losses = train_model(model, config)

    """Visualize initial model"""
    vizualize_training_process(losses)

if __name__ == "__main__":
    main()
