"""Fixtures of the loggingUtils tests."""

import logging
from collections.abc import Iterator

import pytest

from py_libraries.other.loggingUtils import DEFAULT_LEVEL, setGlobalLevel

LOGGER_PREFIX = 'test_loggingUtils'


@pytest.fixture
def logger_name(request: pytest.FixtureRequest) -> str:
    """Give each test its own logger name, under a prefix that is cleaned afterwards.

    Args:
        request: Pytest request of the running test.

    Returns:
        A logger name unique to the test.
    """
    return f'{LOGGER_PREFIX}.{request.node.name}'


@pytest.fixture(autouse=True)
def cleanLoggers() -> Iterator[None]:
    """Undo what a test did to the logging registry.

    The handlers and filters of the test loggers are removed, and the global level
    is set back to its default, so that no test depends on the previous ones.

    Yields:
        Nothing. The cleaning runs after the test.
    """
    yield

    for name, logger in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith(LOGGER_PREFIX) and isinstance(logger, logging.Logger):
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()
            logger.filters.clear()
            logger.setLevel(logging.NOTSET)
            logger.propagate = True

    setGlobalLevel(DEFAULT_LEVEL)
