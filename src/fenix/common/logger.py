"""Modern, Pythonic logging utilities.

This module provides utilities for enhanced logging capabilities including
ANSI escape code removal, stream redirection, and context management.
"""

import logging
import re
import sys
import warnings
from collections.abc import Generator
from contextlib import contextmanager
from logging import Handler, StreamHandler
from typing import Any

# Compiled regex for better performance
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")


def remove_ansi_escape_codes(text: str) -> str:
    """Remove ANSI escape codes from text.

    Args:
        text: Input text that may contain ANSI escape sequences.

    Returns:
        Text with ANSI escape codes removed.

    Example:
        >>> remove_ansi_escape_codes("\x1b[31mRed text\x1b[0m")
        'Red text'
    """
    if not isinstance(text, str):
        return str(text)
    return _ANSI_ESCAPE_PATTERN.sub("", text)


class LoggingMixin:
    """Mixin class providing logging functionality.

    Provides a logger configured with the class name for easy identification
    in log messages.

    Attributes:
        log: Logger instance for this class.

    Example:
        >>> class MyClass(LoggingMixin):
        ...     def do_something(self):
        ...         self.log.info("Doing something")
    """

    def __init__(self, context: Any | None = None) -> None:
        """Initialize the logging mixin.

        Args:
            context: Optional context to set for the logger.
        """
        self._log: logging.Logger | None = None

    @property
    def log(self) -> logging.Logger:
        """Get logger instance for this class.

        Returns:
            Logger instance configured with the class name.
        """
        logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._log = logging.getLogger(logger_name)
        return self._log

