from mlp import MLP
from data_handler import prepare_xor_data, get_xor_training_data, get_xor_test_data, initialize_weights_and_biases
from config_manager import ConfigManager
from training import train_model
from utils.visualizer import vizualize_training_process
from utils.logger import get_logger
import uuid

def main():
    """Initialize config"""
    config = ConfigManager()

    """Logger setup"""
    logger = get_logger("Main", config)
    logger.info("Starting the main process...")
    logger.info(f"Configuration loaded: lr={config.learning_rate}, epochs={config.epoch_count}")
    
    """Prepare XOR data"""
    X, Y = prepare_xor_data()
    logger.info(f"XOR data prepared: {X.shape} inputs, {Y.shape} outputs")

    """Initialize model"""
    model = MLP(X, Y, config)
    logger.info("MLP model initialized")


    """Initialize weights and biases"""

    layer_sizes = config.layer_sizes  # example, [2, 4, 1]
    activations = config.activations  # example, ['Tanh', 'Tanh']

    weights, biases = initialize_weights_and_biases(layer_sizes)

    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    """Create model layers"""

    for i in range(len(layer_sizes) - 1):
        model.create_layer(
            layer_sizes[i],
            layer_sizes[i+1],
            weights[i],
            biases[i],
            activation=activations[i]
        )

    """Log the architecture of the network"""
    arch_str = " -> ".join(str(size) for size in layer_sizes)
    activations_str = ", ".join(activations)
    logger.info(f"Network architecture: {arch_str} ({activations_str})")

    """Prepare training data"""
    training_data = get_xor_training_data()

    """Train model"""
    logger.info("Starting model training...")
    losses = train_model(model, config, training_data)
    logger.info(f"Training completed. Final loss: {losses[-1]:.6f}")

    """Visualize initial model"""
    show_visualization = input("Show training visualization? (y/n): ").strip().lower()
    if show_visualization == "y":
        logger.info("Generating training visualization...")
        vizualize_training_process(losses)
    logger.info("Training process completed successfully")

    model.get_model_info()

    test_model = input("Do you want to test the model? (y/n): ").strip().lower()
    if test_model == "y":
        from utils.testing import test_model
        logger.info("Testing model...")
        test_data = get_xor_test_data()
        test_model(model, test_data)
        logger.info("Model testing completed")

    is_save = input("Do you want to save the model? (y/n): ").strip().lower()
    if is_save == "y":
        unique_id = str(uuid.uuid4())
        default_name = f"{unique_id}.npy"
        model_name = input(f"Enter model name (default is default generated {default_name}): ").strip() or default_name
        model.save_model(model_name)
        logger.info(f"Model saved as {model_name}")

if __name__ == "__main__":
    main()
