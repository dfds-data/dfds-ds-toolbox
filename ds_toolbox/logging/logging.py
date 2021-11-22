import logging
from collections import defaultdict
from pathlib import Path
from typing import Union

from rich.logging import RichHandler

# Richhandler already provides coloured asctime and message level
STANDARD_FORMAT = "%(module)s - %(message)s"
# A default dict is used in case we want to add different formats for different levels
LOG_FORMATS = defaultdict(
    lambda: STANDARD_FORMAT,
    {"DEBUG_FILE": "[%(asctime)s] %(levelname)s %(name)s: %(module)s - %(message)s"},
)


def init_logger(
    name: str = None, stream_level: str = "WARNING", debug_file: Union[Path, str] = None
) -> logging.Logger:
    """Initialize a logger.

    Set up logging that print to stdout with
    ``stream_level`` (default value is WARNING). If ``debug_file`` is
    given set up logging to file with DEBUG level.

    Args:
        name: name of the logger
        stream_level: threshold level for the logging
        debug_file: filename for the logging debug file

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
    """

    # Set up logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Remove all attached handlers, in case there was
    # a logger with using the name
    del logger.handlers[:]
    # Create a file handler if a log file is provided
    if debug_file is not None:
        debug_formatter = logging.Formatter(LOG_FORMATS["DEBUG_FILE"])
        file_handler = logging.FileHandler(filename=debug_file, mode="w")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(debug_formatter)
        logger.addHandler(file_handler)

    log_formatter = logging.Formatter(LOG_FORMATS[stream_level])
    handler = RichHandler(
        level=stream_level, rich_tracebacks=True, log_time_format="[%Y-%m-%d %H:%M:%S,%f]"
    )
    handler.setFormatter(log_formatter)
    logger.addHandler(handler)

    return logger
