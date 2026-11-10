"""Unit tests for getLogger."""

import logging
import re
from pathlib import Path
from typing import cast

import pytest

from py_libraries.other.loggingUtils import (
    DEFAULT_FORMAT,
    DEFAULT_LEVEL,
    getLogger,
    setGlobalLevel,
)

ANSI_PATTERN = re.compile(r'\x1b\[[0-9;]*m')
DEFAULT_DATE_FORMAT_PATTERN = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}'


def test_returns_the_logger_registered_under_the_name(logger_name: str) -> None:
    """Return the same object as `logging.getLogger`.

    Args:
        logger_name: Name of the logger of the test.
    """
    logger = getLogger(logger_name)

    assert isinstance(logger, logging.Logger)
    assert logger.name == logger_name
    assert logger is logging.getLogger(logger_name)


def test_default_level_is_info(logger_name: str) -> None:
    """Use INFO when neither a level nor a global level is given.

    Args:
        logger_name: Name of the logger of the test.
    """
    assert DEFAULT_LEVEL == logging.INFO
    assert getLogger(logger_name).level == logging.INFO


@pytest.mark.parametrize(
    ('level', 'expected'),
    [
        (logging.DEBUG, logging.DEBUG),
        (logging.ERROR, logging.ERROR),
        ('WARNING', logging.WARNING),
        ('CRITICAL', logging.CRITICAL),
        (logging.NOTSET, logging.NOTSET),
    ],
)
def test_explicit_level_is_applied(
    logger_name: str, level: int | str, expected: int
) -> None:
    """Apply a level given as an int or as a level name.

    Args:
        logger_name: Name of the logger of the test.
        level: Level passed to the function.
        expected: Numeric level expected on the logger.
    """
    assert getLogger(logger_name, level=level).level == expected


def test_explicit_level_wins_over_the_global_level(logger_name: str) -> None:
    """Prefer the level of the call to the global level.

    Args:
        logger_name: Name of the logger of the test.
    """
    setGlobalLevel(logging.ERROR)

    assert getLogger(logger_name, level=logging.DEBUG).level == logging.DEBUG


def test_global_level_is_used_when_no_level_is_given(logger_name: str) -> None:
    """Take the level set by `setGlobalLevel`.

    Args:
        logger_name: Name of the logger of the test.
    """
    setGlobalLevel(logging.WARNING)

    assert getLogger(logger_name).level == logging.WARNING


def test_level_is_applied_again_at_each_call(logger_name: str) -> None:
    """Replace the level of an existing logger by the one of the new call.

    Args:
        logger_name: Name of the logger of the test.
    """
    getLogger(logger_name, level=logging.DEBUG)

    assert getLogger(logger_name).level == logging.INFO
    assert getLogger(logger_name, level=logging.ERROR).level == logging.ERROR


def test_logger_does_not_propagate_to_the_root_logger(logger_name: str) -> None:
    """Disable the propagation, which would duplicate the messages.

    Args:
        logger_name: Name of the logger of the test.
    """
    assert getLogger(logger_name).propagate is False


def test_console_handler_writes_to_stdout(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print the messages on stdout and nothing on stderr.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    logger = getLogger(logger_name)

    logger.info('hello')

    captured = capsys.readouterr()
    assert 'hello' in captured.out
    assert captured.err == ''


def test_console_line_follows_the_default_format(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Write level, name, date and message, with a colored level.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    assert DEFAULT_FORMAT == '%(levelname)s - %(name)s - %(asctime)s  - %(message)s'
    logger = getLogger(logger_name)

    logger.info('hello')

    line = capsys.readouterr().out.rstrip('\n')
    plain = ANSI_PATTERN.sub('', line)
    assert re.fullmatch(
        rf'INFO - {re.escape(logger_name)} - {DEFAULT_DATE_FORMAT_PATTERN}  - hello',
        plain,
    )
    assert line.startswith('\x1b[32mINFO\x1b[0m')


@pytest.mark.parametrize(
    ('method', 'color'),
    [
        ('debug', '\x1b[34m'),
        ('info', '\x1b[32m'),
        ('warning', '\x1b[38;5;208m'),
        ('error', '\x1b[31m'),
        ('critical', '\x1b[31;1m'),
    ],
)
def test_each_level_has_its_color(
    logger_name: str,
    capsys: pytest.CaptureFixture[str],
    method: str,
    color: str,
) -> None:
    """Color the level name according to the level of the message.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
        method: Name of the logging method to call.
        color: Expected ANSI sequence.
    """
    logger = getLogger(logger_name, level=logging.DEBUG)

    getattr(logger, method)('hello')

    assert capsys.readouterr().out.startswith(color)


def test_messages_below_the_level_are_dropped(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Print only the messages that reach the level.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    logger = getLogger(logger_name, level='WARNING')

    logger.info('quiet')
    logger.warning('loud')

    out = capsys.readouterr().out
    assert 'quiet' not in out
    assert 'loud' in out


def test_custom_format_and_date_format_are_used(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Apply the given format string and date format to the console line.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    logger = getLogger(
        logger_name, format_str='%(asctime)s|%(message)s', date_format='%Y'
    )

    logger.info('hello')

    assert re.fullmatch(r'\d{4}\|hello\n', capsys.readouterr().out)


def test_second_call_does_not_stack_handlers(
    logger_name: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep a single console handler, so that a message is printed once.

    Args:
        logger_name: Name of the logger of the test.
        capsys: Pytest fixture that captures the output.
    """
    getLogger(logger_name)
    logger = getLogger(logger_name)

    logger.info('hello')

    assert len(logger.handlers) == 1
    assert capsys.readouterr().out.count('hello') == 1


@pytest.mark.parametrize('as_path', [False, True])
def test_log_file_receives_plain_lines(
    logger_name: str, tmp_path: Path, as_path: bool
) -> None:
    """Write the lines to the file without any color, as a str or a Path.

    Args:
        logger_name: Name of the logger of the test.
        tmp_path: Pytest temporary directory.
        as_path: Whether the file is given as a Path instead of a str.
    """
    log_file = tmp_path / 'app.log'
    logger = getLogger(logger_name, log_file=log_file if as_path else str(log_file))

    logger.info('hello')

    content = log_file.read_text()
    assert ANSI_PATTERN.search(content) is None
    assert re.fullmatch(
        rf'INFO - {re.escape(logger_name)} - {DEFAULT_DATE_FORMAT_PATTERN}  - hello\n',
        content,
    )
    assert len(logger.handlers) == 2


def test_log_file_uses_the_custom_format(logger_name: str, tmp_path: Path) -> None:
    """Apply the format string and the date format to the file too.

    Args:
        logger_name: Name of the logger of the test.
        tmp_path: Pytest temporary directory.
    """
    log_file = tmp_path / 'app.log'
    logger = getLogger(
        logger_name,
        log_file=log_file,
        format_str='%(asctime)s|%(message)s',
        date_format='%Y',
    )

    logger.info('hello')

    assert re.fullmatch(r'\d{4}\|hello\n', log_file.read_text())


def test_no_file_handler_without_log_file(logger_name: str) -> None:
    """Add only the console handler when no file is given.

    Args:
        logger_name: Name of the logger of the test.
    """
    logger = getLogger(logger_name)

    assert [type(handler) for handler in logger.handlers] == [logging.StreamHandler]


def test_logger_with_handlers_keeps_them(logger_name: str) -> None:
    """Leave a logger that already has a handler as it is, except for its level.

    Args:
        logger_name: Name of the logger of the test.
    """
    logger = logging.getLogger(logger_name)
    handler = logging.NullHandler()
    logger.addHandler(handler)

    result = getLogger(logger_name, level=logging.ERROR)

    assert result is logger
    assert result.handlers == [handler]
    assert result.propagate is True
    assert result.level == logging.ERROR


def test_log_file_is_ignored_for_a_logger_that_has_handlers(
    logger_name: str, tmp_path: Path
) -> None:
    """Create no file when the logger was already configured.

    Args:
        logger_name: Name of the logger of the test.
        tmp_path: Pytest temporary directory.
    """
    getLogger(logger_name)

    getLogger(logger_name, log_file=tmp_path / 'late.log')

    assert not (tmp_path / 'late.log').exists()


def test_unknown_level_name_raises_value_error(logger_name: str) -> None:
    """Raise ValueError for a string that is not a level name.

    Args:
        logger_name: Name of the logger of the test.
    """
    with pytest.raises(ValueError, match='Unknown level'):
        getLogger(logger_name, level='LOUD')


def test_level_of_a_wrong_type_raises_type_error(logger_name: str) -> None:
    """Raise TypeError for a level that is neither an int nor a string.

    Args:
        logger_name: Name of the logger of the test.
    """
    with pytest.raises(TypeError):
        getLogger(logger_name, level=cast(int, 3.5))


def test_failed_call_does_not_configure_the_logger(logger_name: str) -> None:
    """Leave the logger without handler when the level is rejected.

    Args:
        logger_name: Name of the logger of the test.
    """
    with pytest.raises(ValueError, match='Unknown level'):
        getLogger(logger_name, level='LOUD')

    assert logging.getLogger(logger_name).handlers == []


def test_unwritable_log_file_raises_os_error(logger_name: str, tmp_path: Path) -> None:
    """Raise OSError when the log file cannot be opened.

    Args:
        logger_name: Name of the logger of the test.
        tmp_path: Pytest temporary directory.
    """
    with pytest.raises(OSError):
        getLogger(logger_name, log_file=tmp_path / 'missing_dir' / 'app.log')
