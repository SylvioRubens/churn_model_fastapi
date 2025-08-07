import logging
from logging.handlers import TimedRotatingFileHandler
import os
import sys

def setup_logger():
    """Function to set up a logger that writes logs to a file with daily rotation.
    """
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logger = logging.getLogger(__name__)
    
    logger.setLevel(logging.INFO)  # Set the overall logging level
    
    handler = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'app.log'),
        when='midnight',
        interval=1,
        backupCount=7,  # Keep 7 days of logs
        encoding='utf-8'
    )
    
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(formatter)

    # Redirecionar para stdout (Gunicorn capta isso)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
        
    return logger