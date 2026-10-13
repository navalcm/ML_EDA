"""
Centralized logging configuration for the ML pipeline.

This module provides a reusable logger that writes logs to:
1. A log file
2. The console/terminal
"""

import logging

from src.config import LOG_DIR, LOG_FILE

# ============================================================
# LOGGER CONFIGURATION
# ============================================================

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s"

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# ============================================================
# GET LOGGER
# ============================================================


def get_logger(name: str = "ml_pipeline") -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Name of the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    # --------------------------------------------------------
    # Create logs directory if it does not exist
    # --------------------------------------------------------

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Get logger
    # --------------------------------------------------------

    logger = logging.getLogger(name)

    # --------------------------------------------------------
    # Set logging level
    # --------------------------------------------------------

    logger.setLevel(logging.INFO)

    # --------------------------------------------------------
    # Prevent messages from being sent to the root logger
    # --------------------------------------------------------

    logger.propagate = False

    # --------------------------------------------------------
    # Avoid duplicate handlers
    # --------------------------------------------------------

    if logger.handlers:
        return logger

    # --------------------------------------------------------
    # Create formatter
    # --------------------------------------------------------

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # ========================================================
    # FILE HANDLER
    # ========================================================

    file_handler = logging.FileHandler(filename=LOG_FILE, encoding="utf-8")

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # ========================================================
    # CONSOLE HANDLER
    # ========================================================

    console_handler = logging.StreamHandler()

    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ========================================================
    # ADD HANDLERS
    # ========================================================

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# DEFAULT LOGGER
# ============================================================

logger = get_logger()


# ============================================================
# TEST LOGGER
# ============================================================

if __name__ == "__main__":
    logger.debug("Debug message")
    logger.info("ML pipeline started")
    logger.warning("This is a warning")
    logger.error("This is an error")
    logger.critical("Critical error occurred")
