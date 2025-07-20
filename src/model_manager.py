"""
Simple model save/load utilities
"""
import numpy as np
import os
from utils.logger import get_logger

class ModelManager:
    """Simple model save/load manager"""
    
    def __init__(self, config=None):
        self.config = config
        self.logger = get_logger("ModelManager", config) if config else None
    
    def save_model(self, model, file_path):
        """Save model to file"""
        try:
            # if is provided only filename - save in models/
            if not os.path.dirname(file_path):
                file_path = os.path.join("models", file_path)
            
            # Create dir if not exists
            dir_path = os.path.dirname(file_path)
            if dir_path and not os.path.exists(dir_path):
                os.makedirs(dir_path)
            
            # Save only necessary data like weights, biases, and activation functions
            model_data = {
                'layers': model.layers, 
                'config': model.config.get_all_model_settings() if model.config else None
            }

            np.save(file_path, model_data)
            
            if self.logger:
                self.logger.info(f"Model saved: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Save failed: {e}")
            return False
    
    def load_model(self, model, file_path):
        """Load model from file"""
        try:
            # If only filename is provided, look in models/
            if not os.path.dirname(file_path):
                file_path = os.path.join("models", file_path)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Load model data
            data = np.load(file_path, allow_pickle=True).item()
            
            # Restore layers
            model.layers = data.get('layers', [])
            
            # Restore config if saved
            if 'config' in data and data['config'] is not None:
                # Update model's config with saved values
                if model.config is not None:
                    for key, value in data['config'].items():
                        if hasattr(model.config, key):
                            setattr(model.config, key, value)
            
            if self.logger:
                self.logger.info(f"Model loaded: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Load failed: {e}")
            return False

    def get_current_model_info(self, model):
        """Get current model information"""
        if self.logger:
            self.logger.info(f"Current model info requested")
        
        info = {
            "layers": len(model.layers),
            "layer_details": [],
            "config": model.config.get_all_model_settings() if model.config else None
        }
        
        # Add details about each layer
        for i, layer in enumerate(model.layers):
            layer_info = {
                "layer": i + 1,
                "input_size": layer['input_size'],
                "output_size": layer['output_size'],
                "activation": layer['activation'].__name__ if hasattr(layer['activation'], '__name__') else str(layer['activation'])
            }
            info["layer_details"].append(layer_info)
        
        # Print formatted info
        print("\n" + "="*50)
        print("           CURRENT MODEL INFO")
        print("="*50)
        print(f"Total layers: {info['layers']}")
        print("\nLayer Details:")
        for layer_detail in info["layer_details"]:
            print(f"  Layer {layer_detail['layer']}: {layer_detail['input_size']} -> {layer_detail['output_size']} ({layer_detail['activation']})")
        
        if info['config']:
            print(f"\nConfiguration:")
            print(f"  Learning rate: {info['config'].get('learning_rate', 'N/A')}")
            print(f"  Momentum: {info['config'].get('momentum', 'N/A')}")
            print(f"  Use momentum: {info['config'].get('use_momentum', 'N/A')}")
            print(f"  Epochs: {info['config'].get('epoch_count', 'N/A')}")
            print(f"  First activation: {info['config'].get('first_activation_function_name', 'N/A')}")
            print(f"  Second activation: {info['config'].get('second_activation_function_name', 'N/A')}")
        else:
            print(f"\nConfiguration: No config available")
        print("="*50)
        
        return info

    def get_saved_model_info(self, file_path):
        """Get saved model information"""
        if self.logger:
            self.logger.info(f"Saved model info requested for: {file_path}")
        
        # Ensure we have full path
        if not os.path.dirname(file_path):
            file_path = os.path.join("models", file_path)
        
        if not os.path.exists(file_path):
            print(f"Model file not found: {file_path}")
            return None
        
        try:
            data = np.load(file_path, allow_pickle=True).item()
            layers = data.get('layers', [])
            config = data.get('config', {})
            
            info = {
                "file_path": file_path,
                "layers": len(layers),
                "layer_details": [],
                "config": config
            }
            
            # Add details about each layer
            for i, layer in enumerate(layers):
                layer_info = {
                    "layer": i + 1,
                    "input_size": layer.get('input_size', layer['W'].shape[1]),
                    "output_size": layer.get('output_size', layer['W'].shape[0]),
                    "activation": layer.get('activation_name', layer['activation'].__name__ if hasattr(layer['activation'], '__name__') else 'Unknown')
                }
                info["layer_details"].append(layer_info)
            
            # Print formatted info
            print("\n" + "="*50)
            print("           SAVED MODEL INFO")
            print("="*50)
            print(f"File: {file_path}")
            print(f"Total layers: {info['layers']}")
            print("\nLayer Details:")
            for layer_detail in info["layer_details"]:
                print(f"  Layer {layer_detail['layer']}: {layer_detail['input_size']} -> {layer_detail['output_size']} ({layer_detail['activation']})")
            
            if config:
                print(f"\nConfiguration:")
                print(f"  Learning rate: {config.get('learning_rate', 'N/A')}")
                print(f"  Momentum: {config.get('momentum', 'N/A')}")
                print(f"  Use momentum: {config.get('use_momentum', 'N/A')}")
                print(f"  Epochs: {config.get('epoch_count', 'N/A')}")
                print(f"  First activation: {config.get('first_activation_function_name', 'N/A')}")
                print(f"  Second activation: {config.get('second_activation_function_name', 'N/A')}")
            else:
                print(f"\nConfiguration: No config available")
            print("="*50)
            
            return info
            
        except Exception as e:
            print(f"Error reading model file: {e}")
            if self.logger:
                self.logger.error(f"Get saved model info failed: {e}")
            return None