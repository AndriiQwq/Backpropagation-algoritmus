import os
import configparser
from colorama import Fore, init
init(autoreset=True)

class ConfigManager:
    def __init__(self):
        # Get the current directory and project root
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.config_file = os.path.join(project_root, 'config', 'config.ini')
        self.config = configparser.ConfigParser()
        
        self.default_config = {
            'Settings': {
                'epoch_count': '200',
                'learning_rate': '0.1',
                'momentum': '0.9',
                'use_momentum': 'True',
                'first_activation_function_name(Sigmoid, Tanh, ReLU)': 'Tanh',
                'second_activation_function_name(Sigmoid, Tanh, ReLU)': 'Tanh',
            },
            'Logging': {
                'enable_logging': 'True',
                'log_level': 'INFO'
            }
        }
        
        self.load_config()
    
    def load_config(self):
        if not os.path.exists(self.config_file):
            # Create the config directory if it doesn't exist
            config_dir = os.path.dirname(self.config_file)
            os.makedirs(config_dir, exist_ok=True)

            self.config.read_dict(self.default_config)
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)

            # Log to file if logging is enabled, otherwise print with status
            if self.enable_logging:
                try:
                    from utils.logger import get_logger
                    logger = get_logger("ConfigManager", self)
                    logger.info('Default configuration was created')
                    print(Fore.LIGHTBLUE_EX + 'Default configuration was created' + '\n' + Fore.LIGHTGREEN_EX + "Logger status: Enabled")
                except ImportError:
                    print(Fore.LIGHTBLUE_EX + 'Default configuration was created' + '\n' + Fore.RED + "Logger status: Disabled")
            else:
                print(Fore.LIGHTBLUE_EX + 'Default configuration was created' + '\n' + Fore.RED + "Logger status: Disabled")
        else:
            self.config.read(self.config_file)
            self._log_config_status('Configuration loaded successfully', Fore.LIGHTGREEN_EX)
    
    def _log_config_status(self, message, color=Fore.WHITE):
        """Helper method for logging configuration loading status"""
        try:
            from utils.logger import get_logger
            if self.enable_logging:
                logger = get_logger("ConfigManager", self)
                logger.info(message)
                return
        except ImportError:
            pass
        
        print(color + message + '\n' + (Fore.RED + "Logger status: Disabled") if not self.enable_logging else (Fore.LIGHTGREEN_EX + "Logger status: Enabled"))
        
    @property
    def epoch_count(self):
        return self.config.getint('Settings', 'epoch_count')
    
    @property
    def learning_rate(self):
        return self.config.getfloat('Settings', 'learning_rate')
    
    @property
    def momentum(self):
        return self.config.getfloat('Settings', 'momentum')
    
    @property
    def use_momentum(self):
        return self.config.getboolean('Settings', 'use_momentum')
    
    @property
    def first_activation_function_name(self):
        return self.config.get('Settings', 'first_activation_function_name(Sigmoid, Tanh, ReLU)')
    
    @property
    def second_activation_function_name(self):
        return self.config.get('Settings', 'second_activation_function_name(Sigmoid, Tanh, ReLU)')
    
    @property
    def enable_logging(self):
        return self.config.getboolean('Logging', 'enable_logging', fallback=True)
    
    @property
    def log_level(self):
        return self.config.get('Logging', 'log_level', fallback='INFO')

    def get_all_model_settings(self):
        """Get all model-related settings as dictionary for serialization"""
        return {
            'epoch_count': self.epoch_count,
            'learning_rate': self.learning_rate,
            'momentum': self.momentum,
            'use_momentum': self.use_momentum,
            'first_activation_function_name': self.first_activation_function_name,
            'second_activation_function_name': self.second_activation_function_name
        }
    
    def get_all_settings(self):
        # TODO and add to project
        pass