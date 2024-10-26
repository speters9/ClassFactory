import json
import logging
import time
from functools import wraps
from typing import Any, Callable


def reset_loggers(log_level=logging.WARNING,
                  log_format='%(asctime)s - %(levelname)s - %(message)s - raised_by: %(name)s'):
    # Remove all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set logging level for specific noisy libraries
    noisy_loggers = ['httpx', 'httpcore', 'requests', 'urllib3', 'tqdm', 'transformers', 'langchain_openai', 'gradio']
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(log_level)

    # Set a date format with no milliseconds
    date_format = '%Y-%m-%d %H:%M:%S'

    # Re-apply logging basic config with the new date format
    logging.basicConfig(level=log_level, format=log_format, datefmt=date_format)


def logger_setup(logger_name="query_logger", log_level=logging.INFO):
    """
    Set up and return a logger with the specified name and level.

    Args:
        logger_name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Setup the logger object
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if this function is called multiple times
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()

        # Set the format for logging
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - raised_by: %(name)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    # Set the logger level
    logger.setLevel(log_level)

    return logger


def retry_on_json_decode_error(max_retries: int = 3, delay: float = 2.0):
    """
    Decorator to retry a function if a JSONDecodeError or ValueError is encountered.

    Args:
        max_retries (int): The maximum number of retries.
        delay (float): The delay in seconds between retries.

    Returns:
        Callable: The decorated function with retry logic.
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            func_logger = kwargs.get('logger', logging.getLogger(__name__))

            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except (json.JSONDecodeError, ValueError) as e:
                    attempts += 1
                    func_logger.error(f"Error encountered. Attempt {attempts}/{max_retries}. Error: {e}")
                    if attempts < max_retries:
                        func_logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        func_logger.error("Max retries reached. Raising the exception.")
                        raise
        return wrapper
    return decorator
