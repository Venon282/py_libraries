"""Unit tests for ColoredFormatter.format."""

import logging
import sys

import pytest

from py_libraries.other.loggingUtils import ColoredFormatter

RESET = '\x1b[0m'


def makeRecord(
    level: int, message: str = 'hello', args: tuple[object, ...] | None = None
) -> logging.LogRecord:
    """Build a record of a given level.

    Args:
        level: Numeric level of the record.
        message: Message, possibly with `%` placeholders.
        args: Arguments of the message placeholders.

    Returns:
        The record.
    """
    return logging.LogRecord('name', level, __file__, 1, message, args, None)


@pytest.mark.parametrize(
    ('level', 'color'),
    [
        (logging.DEBUG, '\x1b[34m'),
        (logging.INFO, '\x1b[32m'),
        (logging.WARNING, '\x1b[38;5;208m'),
        (logging.ERROR, '\x1b[31m'),
        (logging.CRITICAL, '\x1b[31;1m'),
    ],
)
def test_level_name_is_wrapped_in_its_color(level: int, color: str) -> None:
    """Surround the level name with its color and the reset sequence.

    Args:
        level: Numeric level of the record.
        color: Expected ANSI sequence of the level.
    """
    formatter = ColoredFormatter('%(levelname)s|%(message)s')

    line = formatter.format(makeRecord(level))

    assert line == f'{color}{logging.getLevelName(level)}{RESET}|hello'


def test_only_the_level_name_is_colored() -> None:
    """Leave the rest of the line free of escape sequences."""
    formatter = ColoredFormatter('%(levelname)s - %(name)s - %(message)s')

    line = formatter.format(makeRecord(logging.INFO))

    assert line.endswith(f'{RESET} - name - hello')
    assert line.count('\x1b') == 2


def test_level_without_color_is_wrapped_with_the_reset_sequence() -> None:
    """Keep a custom level readable with the reset sequence on both sides."""
    record = makeRecord(25)
    record.levelname = 'NOTICE'
    formatter = ColoredFormatter('%(levelname)s|%(message)s')

    assert formatter.format(record) == f'{RESET}NOTICE{RESET}|hello'


@pytest.mark.parametrize('level', [logging.DEBUG, logging.INFO, logging.CRITICAL])
def test_record_is_left_unchanged(level: int) -> None:
    """Restore the level name, so that other handlers get a plain one.

    Args:
        level: Numeric level of the record.
    """
    record = makeRecord(level)
    formatter = ColoredFormatter('%(levelname)s|%(message)s')

    formatter.format(record)

    assert record.levelname == logging.getLevelName(level)


def test_other_formatter_receives_the_plain_level_name() -> None:
    """Give a plain level name to a formatter that runs after the colored one."""
    record = makeRecord(logging.ERROR)

    ColoredFormatter('%(levelname)s').format(record)

    assert logging.Formatter('%(levelname)s').format(record) == 'ERROR'


def test_record_is_restored_even_when_the_formatting_fails() -> None:
    """Restore the level name when the message cannot be built."""
    record = makeRecord(logging.INFO, message='%d', args=('text',))
    formatter = ColoredFormatter('%(levelname)s|%(message)s')

    with pytest.raises(TypeError):
        formatter.format(record)

    assert record.levelname == 'INFO'


def test_message_arguments_are_interpolated() -> None:
    """Build the message from its placeholders and arguments."""
    record = makeRecord(logging.INFO, message='%s has %d items', args=('list', 3))
    formatter = ColoredFormatter('%(message)s')

    assert formatter.format(record) == 'list has 3 items'


def test_exception_traceback_is_appended() -> None:
    """Append the traceback of an exception to the line."""
    formatter = ColoredFormatter('%(message)s')
    try:
        raise ValueError('broken')
    except ValueError:
        record = logging.LogRecord(
            'name', logging.ERROR, __file__, 1, 'failed', None, sys.exc_info()
        )

    line = formatter.format(record)

    assert line.startswith('failed\nTraceback')
    assert line.endswith('ValueError: broken')
