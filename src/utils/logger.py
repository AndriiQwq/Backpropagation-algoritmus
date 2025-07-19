import logging
import os
from datetime import datetime

class Logger:    
    def __init__(self, name="MLProject", config=None):
        self.config = config
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Only setup logger if enabled
        if self.config.enable_logging:
            self.setup_logger()
    
    def setup_logger(self):        
        # Create logs directory
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Get log level from config
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Formatter to simle log lines
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # File handler for all logs (always enabled)
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # Main log file
        file_handler = logging.FileHandler(
            os.path.join(logs_dir, f'training_{timestamp}.log'),
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def get_logger(self):
        """Get configured logger instance"""
        return self.logger


def get_logger(name="MLProject", config=None):
    """Factory function to get logger"""
    logger = Logger(name, config)
    return logger.get_logger()
