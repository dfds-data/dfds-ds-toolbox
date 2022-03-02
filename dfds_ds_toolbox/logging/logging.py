import logging
from pathlib import Path
from typing import Union

from rich.logging import RichHandler

# Rich handler already provides coloured asctime and message level
LOG_FORMATS = {
    "STANDARD_FORMAT": "[%(asctime)s] %(levelname)s %(name)s: %(module)s - %(message)s",
    "RICH_FORMAT": "%(module)s - %(message)s",
}


def init_logger(
    name: str = None,
    stream_level: str = "WARNING",
    debug_file: Union[Path, str] = None,
    rich_handler_enabled: bool = True,
    log_format: str = None,
) -> logging.Logger:
    """Initialize a logger.

    Set up logging that print to stdout with
    ``stream_level`` (default value is WARNING). If ``debug_file`` is
    given set up logging to file with DEBUG level.

    Args:
        name: name of the logger
        stream_level: threshold level for the logging
        debug_file: filename for the logging debug file
        rich_handler_enabled: enable rich handler with coloured text
        log_format: custom log format

    Returns:
        The `Logger` object.

    Examples:
        Initializing the logger:

        >>> logger = init_logger()
        >>> logger.info("This message will not be logged.")
        >>> logger.critical("Something BAD happened.")

        If you want to log the messages on a different level than DEBUG (default) use:

        >>> logger = init_logger(stream_level="INFO")
        >>> logger.debug("This message will not be logged.")
        >>> logger.info("Starting some work.")

        If you want to save additionally the log the messages into a file use:

        >>> logger = init_logger(stream_level="INFO", debug_file="path/log_files.log")
        >>> logger.debug("Logging something to a file.")
        >>> logger.info("Logging something to both terminal and file.")

        If you do not want to use coloured text in the log use:
        >>> logger = init_logger(stream_level="INFO", rich_handler_enabled=False)
        >>> logger.info("Not coloured text.")

        If you want to use a custom log format use:
        >>> logger = init_logger(stream_level="INFO", log_format="%(module)s - %(message)s", rich_handler_enabled=False)
        >>> logger.info("Custom log format.")

    """

    # Set up logger
    logger = logging.getLogger(name)
    # Setting logger root level (it could be override by specific handler levels)
    logger.setLevel(logging.DEBUG)
    if rich_handler_enabled and log_format is not None:
        raise ValueError("Rich handler and custom log format cannot be used together")
    if log_format is None:
        log_format = LOG_FORMATS["STANDARD_FORMAT"]

    # Remove all attached handlers, in case there was
    # a logger with using the name
    del logger.handlers[:]
    # Create a file handler if a log file is provided
    if debug_file is not None:
        debug_formatter = logging.Formatter(log_format)
        file_handler = logging.FileHandler(filename=debug_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_formatter)
        logger.addHandler(file_handler)
    if rich_handler_enabled:
        # If rich handler is enabled, it comes with specific format so we dont allow custom formatter
        handler = RichHandler(
            level=stream_level, rich_tracebacks=True, log_time_format="[%Y-%m-%d %H:%M:%S,%f]"
        )

        handler.setFormatter(logging.Formatter(LOG_FORMATS["RICH_FORMAT"]))
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(log_format))
        handler.setLevel(level=stream_level)
    logger.addHandler(handler)

    return logger
