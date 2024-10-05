import logging


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
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    # Set the logger level
    logger.setLevel(log_level)

    return logger
