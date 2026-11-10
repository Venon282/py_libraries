"""Unit tests for setGlobalLevel."""

import logging
from typing import cast

import pytest

from py_libraries.other.loggingUtils import getLogger, setGlobalLevel


def test_existing_project_loggers_get_the_level(logger_name: str) -> None:
    """Change the level of every logger created by `getLogger`.

    Args:
        logger_name: Name of the logger of the test.
    """
    first = getLogger(f'{logger_name}.first')
    second = getLogger(f'{logger_name}.second', level=logging.DEBUG)

    setGlobalLevel(logging.ERROR)

    assert first.level == logging.ERROR
    assert second.level == logging.ERROR


def test_future_project_loggers_get_the_level(logger_name: str) -> None:
    """Give the level to the loggers created after the call.

    Args:
        logger_name: Name of the logger of the test.
    """
    setGlobalLevel(logging.CRITICAL)

    assert getLogger(logger_name).level == logging.CRITICAL


def test_level_can_be_given_as_a_name(logger_name: str) -> None:
    """Accept a level name, for existing and for future loggers.

    Args:
        logger_name: Name of the logger of the test.
    """
    existing = getLogger(f'{logger_name}.existing')

    setGlobalLevel('WARNING')

    assert existing.level == logging.WARNING
    assert getLogger(f'{logger_name}.future').level == logging.WARNING


def test_last_call_wins(logger_name: str) -> None:
    """Keep the level of the latest call.

    Args:
        logger_name: Name of the logger of the test.
    """
    logger = getLogger(logger_name)

    setGlobalLevel(logging.ERROR)
    setGlobalLevel(logging.DEBUG)

    assert logger.level == logging.DEBUG
    assert getLogger(f'{logger_name}.future').level == logging.DEBUG


def test_level_of_a_filtered_message_follows_the_global_level(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Hide the messages below the new level and show the others.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    logger = getLogger(logger_name)

    setGlobalLevel(logging.ERROR)
    logger.warning('quiet')
    logger.error('loud')

    out = capsys.readouterr().out
    assert 'quiet' not in out
    assert 'loud' in out


def test_loggers_not_created_by_getLogger_are_untouched(logger_name: str) -> None:
    """Leave the other loggers of the process alone.

    Args:
        logger_name: Name of the logger of the test.
    """
    foreign = logging.getLogger(f'{logger_name}.foreign')

    setGlobalLevel(logging.ERROR)

    assert foreign.level == logging.NOTSET


def test_logger_that_had_handlers_is_still_tracked(logger_name: str) -> None:
    """Change the level of a logger that `getLogger` returned without configuring.

    Args:
        logger_name: Name of the logger of the test.
    """
    logger = logging.getLogger(logger_name)
    logger.addHandler(logging.NullHandler())
    getLogger(logger_name)

    setGlobalLevel(logging.CRITICAL)

    assert logger.level == logging.CRITICAL


def test_notset_level_is_applied_to_existing_and_future_loggers(
    logger_name: str,
) -> None:
    """Apply NOTSET like any other level.

    Args:
        logger_name: Name of the logger of the test.
    """
    existing = getLogger(f'{logger_name}.existing')

    setGlobalLevel(logging.NOTSET)

    assert existing.level == logging.NOTSET
    assert getLogger(f'{logger_name}.future').level == logging.NOTSET


def test_unknown_level_name_raises_value_error() -> None:
    """Raise ValueError for a string that is not a level name."""
    with pytest.raises(ValueError, match='Unknown level'):
        setGlobalLevel('LOUD')


def test_level_of_a_wrong_type_raises_type_error() -> None:
    """Raise TypeError for a level that is neither an int nor a string."""
    with pytest.raises(TypeError):
        setGlobalLevel(cast(int, 3.5))


@pytest.mark.parametrize('invalid', ['LOUD', cast(int, 3.5)])
def test_rejected_level_changes_nothing(logger_name: str, invalid: int | str) -> None:
    """Keep the level of existing and future loggers when the level is rejected.

    Args:
        logger_name: Name of the logger of the test.
        invalid: Level that the function rejects.
    """
    setGlobalLevel(logging.WARNING)
    existing = getLogger(f'{logger_name}.existing')

    with pytest.raises((ValueError, TypeError)):
        setGlobalLevel(invalid)

    assert existing.level == logging.WARNING
    assert getLogger(f'{logger_name}.future').level == logging.WARNING


def test_rejected_first_call_keeps_the_default_level(logger_name: str) -> None:
    """Keep INFO for the new loggers when the very first call is rejected.

    Args:
        logger_name: Name of the logger of the test.
    """
    with pytest.raises(ValueError, match='Unknown level'):
        setGlobalLevel('LOUD')

    assert getLogger(logger_name).level == logging.INFO
