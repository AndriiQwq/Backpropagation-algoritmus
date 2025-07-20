from mlp import MLP
from data_handler import prepare_xor_data, initialize_weights_and_biases
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

    hidden_layer_size = 4
    W1, W2, B1, B2 = initialize_weights_and_biases(2, hidden_layer_size)
    logger.debug(f"Weights initialized: W1{W1.shape}, W2{W2.shape}")

    """
                0
            0   0   
    (input) ->  ->  0 (output)  
            0   0   
                0
    """

    model.create_layer(2, hidden_layer_size, W1, B1, activation=config.first_activation_function_name)
    model.create_layer(hidden_layer_size, 1, W2, B2, activation=config.second_activation_function_name)
    logger.info(f"Network architecture: 2 -> {hidden_layer_size} -> 1 ({config.first_activation_function_name}, {config.second_activation_function_name})")

    """Train model"""
    logger.info("Starting model training...")
    losses = train_model(model, config)
    logger.info(f"Training completed. Final loss: {losses[-1]:.6f}")

    """Visualize initial model"""
    show_visualization = input("Show training visualization? (y/n): ").strip().lower()
    if show_visualization == "y":
        logger.info("Generating training visualization...")
        vizualize_training_process(losses)
    logger.info("Training process completed successfully")

    model.get_model_info()

    is_save = input("Do you want to save the model? (y/n): ").strip().lower()
    if is_save == "y":
        unique_id = str(uuid.uuid4())
        default_name = f"{unique_id}.npy"
        model_name = input(f"Enter model name (default is default generated {default_name}): ").strip() or default_name
        model.save_model(model_name)
        logger.info(f"Model saved as {model_name}")

if __name__ == "__main__":
    main()
