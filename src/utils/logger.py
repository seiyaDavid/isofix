"""
Logging Configuration Module

This module provides a consistent logging setup across the application.
Features:
    - Configurable log levels
    - File and console output
    - Consistent formatting
    - Automatic log rotation
    - Named loggers for different components
"""

from loguru import logger
import sys
from pathlib import Path


def setup_logger(name: str) -> logger:
    """
    Setup a configured logger instance with file and console handlers.

    Args:
        name (str): Name of the logger/component for identification

    Returns:
        logger: Configured loguru logger instance with:
            - File output with daily rotation
            - Console output
            - Consistent formatting
            - Component name tracking

    Example:
        >>> logger = setup_logger("my_component")
        >>> logger.info("Component started")
        2024-01-20 10:30:45 | INFO | my_component | Component started
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    # Configure logger
    logger.remove()  # Remove default handler

    # Add file handler
    logger.add(
        f"logs/{name}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="INFO",
        rotation="1 day",
    )

    # Add console handler
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        level="INFO",
    )

    return logger.bind(name=name)
