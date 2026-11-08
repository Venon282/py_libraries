import logging
import sys
from typing import Optional

_current_level: int | str = logging.INFO
_project_loggers: set[str] = set()

class ColoredFormatter(logging.Formatter):
    """
    Custom Formatter to add colors to the levelname based on the log level.
    """

    # ANSI Escape Sequences
    BLUE = "\x1b[34m"
    GREEN = "\x1b[32m"
    ORANGE = "\x1b[38;5;208m"  # 8-bit color for true orange
    RED = "\x1b[31m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    # Mapping levels to your requested colors
    COLORS = {
        logging.DEBUG: BLUE,
        logging.INFO: GREEN,
        logging.WARNING: ORANGE,
        logging.ERROR: RED,
        logging.CRITICAL: BOLD_RED,
    }

    def __init__(self, fmt: str):
        super().__init__()
        self.fmt = fmt
        self._plain_formatter = logging.Formatter(self.fmt, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record):
        # Apply color to the levelname
        level_color = self.COLORS.get(record.levelno, self.RESET)

        # Save original levelname to restore it later (so file logs stay clean)
        orig_levelname = record.levelname
        record.levelname = f"{level_color}{orig_levelname}{self.RESET}"

        result = self._plain_formatter.format(record)

        # Restore original levelname for other handlers (like file logging)
        record.levelname = orig_levelname
        return result

def setGlobalLevel(level: int | str) -> None:
    """Set the level of every project logger, existing and future ones."""
    global _current_level
    _current_level = level
    for name in _project_loggers:
        logging.getLogger(name).setLevel(level)


def getLogger(
    name: str,
    level: int | str | None = None,
    log_file: str | None = None,
    format_str: str = "%(levelname)s - %(name)s - %(asctime)s  - %(message)s",
) -> logging.Logger:
    logger = logging.getLogger(name)
    _project_loggers.add(name)

    # Explicit level wins, otherwise use the current global level.
    logger.setLevel(level if level is not None else _current_level)

    # root logger and stays True if a library called basicConfig().
    if logger.handlers:
        return logger

    logger.propagate = False  # avoid duplicated messages through the root logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColoredFormatter(format_str))
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(format_str, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    return logger