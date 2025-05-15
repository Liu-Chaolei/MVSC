import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger instance with proper formatting and configuration.

    Args:
        name: The name of the logger, typically __name__ from the calling module
        level: Optional logging level. If not provided, defaults to INFO

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)

    # If logger already has handlers, return it to avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set default level to INFO if not specified
    if level is None:
        level = logging.INFO

    logger.setLevel(level)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter with color support
    class ColoredFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                record.levelname = f"\033[32m{record.levelname}\033[0m"  # Green color for INFO
                record.msg = f"\033[32m{record.msg}\033[0m"  # Green color for message
            elif record.levelno in (logging.WARNING, logging.ERROR):
                record.levelname = (
                    f"\033[31m{record.levelname}\033[0m"  # Red color for WARNING and ERROR
                )
                record.msg = f"\033[31m{record.msg}\033[0m"  # Red color for message
            return super().format(record)

    formatter = ColoredFormatter("\n- %(levelname)s -\n%(message)s\n")
    console_handler.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(console_handler)

    return logger
