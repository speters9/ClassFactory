import json
import logging
import os
import time
import unicodedata
from functools import wraps
from typing import Any, Callable

from langchain_core.exceptions import OutputParserException


def reset_loggers(log_level=logging.WARNING,
                  log_format='%(asctime)s - %(levelname)s - %(message)s - raised_by: %(name)s'):
    # Remove all existing handlers from the root logger to start fresh
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Apply a basic config to the root logger
    logging.basicConfig(level=log_level, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')

    # Set specific noisy libraries to a higher log level (e.g., WARNING)
    noisy_loggers = ['httpx', 'httpcore', 'requests', 'urllib3', 'tqdm', 'transformers', 'langchain_openai', 'gradio', 'sentence_transformers']
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)  # Adjust to WARNING if INFO is too verbose


def logger_setup(logger_name="query_logger", log_level=logging.INFO):
    """
    Set up and return a logger with the specified name and level.
    Avoids affecting the root logger by setting propagate to False.

    Args:
        logger_name (str): The name of the logger.
        log_level (int): The logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
    # Retrieve or create a logger
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if already set up
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)  # Match handler level to logger level

        # Set the format for the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - raised_by: %(name)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    # Set the logger level explicitly and prevent it from propagating to the root
    logger.setLevel(log_level)
    logger.propagate = True

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
                except (json.JSONDecodeError, ValueError, OutputParserException) as e:
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


def print_directory_tree(path, level=0):
    """Recursively formats the directory structure in a Markdown-friendly way."""
    markdown_str = ""
    indent = "    " * level  # Indentation for subdirectories
    markdown_str += f"{indent}- **{path.name}/**\n"  # Bold for directories
    for child in path.iterdir():
        if child.is_dir():
            markdown_str += print_directory_tree(child, level + 1)
        else:
            markdown_str += f"{indent}    - {child.name}\n"
    return markdown_str


def normalize_unicode(text: str) -> str:
    """
    Normalize unicode text to NFKC form (standardizes characters, removes most oddities).
    """
    if not isinstance(text, str):
        return text
    return unicodedata.normalize('NFKC', text)
