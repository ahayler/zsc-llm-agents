import logging

# Basic configuration of logging:
logging.basicConfig(
    level=logging.INFO,  # Set the threshold for the logger
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Get a logger instance (using __name__ helps identify where the log is coming from)
logger = logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)