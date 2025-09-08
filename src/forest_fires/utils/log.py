"""Define a reusable logger."""

import logging


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a configured logger instance.

    Args:
        name: The name of the logger.

    Returns:
        A configured logger instance.
    """
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:  # configure once, only if not already set
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    return logger
