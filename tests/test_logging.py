import logging

import pytest
from rich.logging import RichHandler

from dfds_ds_toolbox.logging.logging import init_logger


@pytest.fixture
def default_logger():
    """Fixture. Call logger setup with WARNING level(default level)."""
    return init_logger()


@pytest.fixture
def info_logger():
    """Fixture. Call logger setup with INFO level."""
    return init_logger(stream_level="INFO")


@pytest.fixture
def debug_logger():
    """Fixture. Call logger setup with DEBUG level."""
    return init_logger(stream_level="DEBUG")


@pytest.fixture
def debug_file_info_logger(debug_file):
    """Fixture. Call info logger setup with debug file."""
    return init_logger(debug_file=debug_file, stream_level="INFO")


@pytest.fixture
def debug_file(tmp_path):
    """Fixture. Generate debug file location for tests."""
    return tmp_path.joinpath("pytest-plugin.log")


def test_instance_logger():
    """Test we can create instance of logger."""
    logger = init_logger()
    assert isinstance(logger, logging.Logger)


def test_instance_for_stream_levels():
    for stream_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        logger = init_logger(stream_level=stream_level)
        assert isinstance(logger, logging.Logger)


def test_debug_level(debug_logger):
    """Test the handle level is set to DEBUG"""
    [rich_handler] = [
        handler for handler in debug_logger.handlers if isinstance(handler, RichHandler)
    ]
    assert rich_handler.level == logging.DEBUG
    assert isinstance(rich_handler, RichHandler)


def test_info_level(info_logger):
    """Test the handle level is set to INFO"""
    [rich_handler] = [
        handler for handler in info_logger.handlers if isinstance(handler, RichHandler)
    ]
    assert rich_handler.level == logging.INFO
    assert isinstance(rich_handler, RichHandler)


def test_default_level(default_logger):
    """Test the handle level is set to WARNING (default value)"""
    [rich_handler] = [
        handler for handler in default_logger.handlers if isinstance(handler, RichHandler)
    ]
    assert rich_handler.level == logging.WARNING
    assert isinstance(rich_handler, RichHandler)


def test_debug_file_logging(debug_file_info_logger, debug_file):
    """Test that logging to stdout uses a different format and level than \
    the the file handler."""

    [file_handler, rich_handler] = [
        handler
        for handler in debug_file_info_logger.handlers
        if (isinstance(handler, RichHandler) or isinstance(handler, logging.FileHandler))
    ]
    assert isinstance(file_handler, logging.FileHandler)
    assert isinstance(rich_handler, RichHandler)
    assert rich_handler.level == logging.INFO
    assert file_handler.level == logging.DEBUG

    debug_file_info_logger.info("Text in log file.")
    assert debug_file.exists()


def test_wrong_args():
    """Test that we get KeyError when specifying a unsupported stream level"""
    with pytest.raises(ValueError):
        init_logger(stream_level="INF")
