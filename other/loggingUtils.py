"""Console and file logging helpers shared by the modules of the project."""

import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

DEFAULT_LEVEL: int = logging.INFO
DEFAULT_FORMAT = '%(levelname)s - %(name)s - %(asctime)s  - %(message)s'
DEFAULT_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# The global level is stored in the logging registry, as the level of a dedicated
# logger, instead of a module variable: the module itself holds no mutable state.
_GLOBAL_LEVEL_LOGGER_NAME = f'{__name__}.globalLevel'


class ColoredFormatter(logging.Formatter):
    """Formatter that wraps the level name of each record in an ANSI color.

    DEBUG is blue, INFO green, WARNING orange, ERROR red and CRITICAL bold red. A
    custom level without a color is wrapped with the reset sequence, which leaves
    it uncolored.

    Args:
        fmt: Format string of the log lines, in the `%` style.
        datefmt: Format of the `asctime` field, in the `time.strftime` style.
    """

    # ANSI escape sequences
    BLUE = '\x1b[34m'
    GREEN = '\x1b[32m'
    ORANGE = '\x1b[38;5;208m'  # 8-bit color for a true orange
    RED = '\x1b[31m'
    BOLD_RED = '\x1b[31;1m'
    RESET = '\x1b[0m'

    COLORS: ClassVar[Mapping[int, str]] = MappingProxyType(
        {
            logging.DEBUG: BLUE,
            logging.INFO: GREEN,
            logging.WARNING: ORANGE,
            logging.ERROR: RED,
            logging.CRITICAL: BOLD_RED,
        }
    )

    def __init__(self, fmt: str, datefmt: str = DEFAULT_DATE_FORMAT) -> None:
        super().__init__(fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format a record with a colored level name.

        The record is left unchanged once formatted, so that the other handlers of
        the logger (a file for example) receive a plain level name.

        Args:
            record: Record to format.

        Returns:
            The formatted log line.
        """
        level_color = self.COLORS.get(record.levelno, self.RESET)

        original_levelname = record.levelname
        record.levelname = f'{level_color}{original_levelname}{self.RESET}'
        try:
            return super().format(record)
        finally:
            record.levelname = original_levelname


class _ProjectLoggerMarker(logging.Filter):
    """Filter that accepts every record and only flags a logger as project logger."""


def _globalLevelLogger() -> logging.Logger:
    """Return the logger whose level is the global level, creating it if needed.

    Returns:
        The logger, initialized with `DEFAULT_LEVEL` when it is created.
    """
    existing = logging.Logger.manager.loggerDict.get(_GLOBAL_LEVEL_LOGGER_NAME)
    if isinstance(existing, logging.Logger):
        return existing

    holder = logging.getLogger(_GLOBAL_LEVEL_LOGGER_NAME)
    holder.setLevel(DEFAULT_LEVEL)
    return holder


def _isProjectLogger(logger: logging.Logger) -> bool:
    """Check whether a logger was configured by `getLogger`.

    Args:
        logger: Logger to inspect.

    Returns:
        True if the logger carries the project marker.
    """
    return any(isinstance(item, _ProjectLoggerMarker) for item in logger.filters)


def _projectLoggers() -> list[logging.Logger]:
    """List the loggers configured by `getLogger`.

    Returns:
        Every registered logger that carries the project marker.
    """
    return [
        logger
        for logger in list(logging.Logger.manager.loggerDict.values())
        if isinstance(logger, logging.Logger) and _isProjectLogger(logger)
    ]


def setGlobalLevel(level: int | str) -> None:
    """Set the level of every project logger, existing and future ones.

    The project loggers are the ones returned by `getLogger`. An invalid level
    changes nothing.

    Args:
        level: Level as an int (`logging.DEBUG`) or as a level name (`'DEBUG'`).

    Raises:
        ValueError: If `level` is a string that is not a level name.
        TypeError: If `level` is neither an int nor a string.
    """
    # Validates the level before any logger is modified
    _globalLevelLogger().setLevel(level)

    for logger in _projectLoggers():
        logger.setLevel(level)


def getLogger(
    name: str,
    level: int | str | None = None,
    log_file: str | Path | None = None,
    format_str: str = DEFAULT_FORMAT,
    date_format: str = DEFAULT_DATE_FORMAT,
) -> logging.Logger:
    """Return a logger that prints colored lines to stdout, and optionally to a file.

    The level is applied at each call. A logger that already has handlers keeps
    them: no handler is added, so `log_file`, `format_str` and `date_format` only
    take effect on the first call for a given name.

    Args:
        name: Name of the logger, usually `__name__`.
        level: Level of the logger, as an int or as a level name. If None, the
            global level set by `setGlobalLevel` is used, which is `DEFAULT_LEVEL`
            until it is set.
        log_file: Path of a file that also receives the log lines, without colors.
            If None, nothing is written to a file.
        format_str: Format string of the log lines, in the `%` style.
        date_format: Format of the `asctime` field, in the `time.strftime` style.

    Returns:
        The configured logger. It does not propagate to the root logger, which
        avoids duplicated messages.

    Raises:
        ValueError: If `level` is a string that is not a level name.
        TypeError: If `level` is neither an int nor a string.
        OSError: If `log_file` cannot be opened.
    """
    logger = logging.getLogger(name)

    # An explicit level wins, otherwise use the current global level
    logger.setLevel(level if level is not None else _globalLevelLogger().level)

    if not _isProjectLogger(logger):
        logger.addFilter(_ProjectLoggerMarker())

    # A logger that already has handlers was configured by an earlier call
    if logger.handlers:
        return logger

    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(format_str, date_format))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt=date_format))
        logger.addHandler(file_handler)

    return logger
