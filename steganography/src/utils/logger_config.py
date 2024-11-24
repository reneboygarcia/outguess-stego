import logging
import os
from datetime import datetime

def setup_logger(name='outguess'):
    """Configure logging to both file and console with different levels"""
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/{name}_{timestamp}.log'
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(message)s'
    )
    
    # File handler (detailed, debug level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler (simple, info level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging to file: {log_file}")
    return logger

def get_logger(name='outguess'):
    """Get or create logger"""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only setup if not already configured
        logger = setup_logger(name)
    return logger 