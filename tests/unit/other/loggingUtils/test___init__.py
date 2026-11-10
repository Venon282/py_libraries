"""Unit tests for ColoredFormatter.__init__."""

import logging
import re

from py_libraries.other.loggingUtils import DEFAULT_DATE_FORMAT, ColoredFormatter

# Fixed in the middle of a year, so that no time zone changes the year
CREATED = 1_000_000_000.0


def makeRecord() -> logging.LogRecord:
    """Build a record whose creation time is fixed.

    Returns:
        An INFO record with the message `hello`.
    """
    record = logging.LogRecord('name', logging.INFO, __file__, 1, 'hello', None, None)
    record.created = CREATED
    return record


def test_default_date_format_is_the_module_constant() -> None:
    """Use `DEFAULT_DATE_FORMAT` when no date format is given."""
    formatter = ColoredFormatter('%(message)s')

    assert formatter.datefmt == DEFAULT_DATE_FORMAT


def test_default_date_format_gives_a_date_and_a_time() -> None:
    """Write the creation time as `YYYY-MM-DD HH:MM:SS`."""
    formatter = ColoredFormatter('%(asctime)s')

    line = formatter.format(makeRecord())

    assert re.fullmatch(r'2001-09-\d{2} \d{2}:\d{2}:\d{2}', line)


def test_custom_date_format_is_used() -> None:
    """Apply the date format that was given."""
    formatter = ColoredFormatter('%(asctime)s', '%Y')

    assert formatter.datefmt == '%Y'
    assert formatter.format(makeRecord()) == '2001'


def test_date_format_can_be_passed_by_keyword() -> None:
    """Accept the date format as a keyword argument."""
    formatter = ColoredFormatter('%(asctime)s', datefmt='%Y')

    assert formatter.format(makeRecord()) == '2001'


def test_format_string_is_used() -> None:
    """Build the line from the given format string."""
    formatter = ColoredFormatter('[%(name)s] %(message)s')

    assert formatter.format(makeRecord()) == '[name] hello'
