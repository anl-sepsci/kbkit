"""Custom logging setup for reusability across kbkit."""

import logging


def get_logger(name: str, verbose: bool = False) -> logging.Logger:
    """
    Create and configure a named logger with stream output and verbosity control.

    Parameters
    ----------
    name : str
        Name of the logger, typically the module or class name.
    verbose : bool, optional
        If True, sets logging level to DEBUG; otherwise WARNING.

    Returns
    -------
    logging.Logger
        Configured logger instance with stream handler and standardized format.

    Notes
    -----
    - Prevents duplicate handlers by checking `logger.handlers`.
    - Format: "[LEVEL] name: message"
    - Intended for use in modules that benefit from consistent, minimal logging.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG if verbose else logging.WARNING)
    return logger
