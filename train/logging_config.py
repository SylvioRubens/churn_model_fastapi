import logging
import os
from logging.handlers import RotatingFileHandler

def setup_train_logging(log_path: str = "logs/training.log") -> logging.Logger:
    """Function to set up logging for the training process"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Clear existing handlers to avoid duplicate logs
    logger.propagate = False  # Prevent logs from being propagated to the root logger
    
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10*1024*1024,  # 5 MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
