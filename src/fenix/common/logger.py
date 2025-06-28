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
        if context is not None:
            self._set_context(context)

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance (deprecated).

        Returns:
            Logger instance.

        Warning:
            This property is deprecated. Use 'log' instead.
        """
        warnings.warn(
            f"Using 'logger' property in {self.__class__.__module__}.{self.__class__.__name__} "
            "is deprecated. Use 'log' property instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.log

    @property
    def log(self) -> logging.Logger:
        """Get logger instance for this class.

        Returns:
            Logger instance configured with the class name.
        """
        if self._log is None:
            logger_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
            self._log = logging.getLogger(logger_name)
        return self._log

    def _set_context(self, context: Any) -> None:
        """Set context for the logger.

        Args:
            context: Context value to set for logger handlers.
        """
        set_logger_context(self.log, context)


class StreamLogWriter:
    """Writer that redirects stream output to a logger.

    This class allows redirecting stdout/stderr to a logger with a specified
    log level, while handling buffering and ANSI escape code removal.

    Attributes:
        encoding: Stream encoding (None for compatibility).
        closed: Always False as the stream remains open.
    """

    encoding: str | None = None

    def __init__(self, logger: logging.Logger, level: int) -> None:
        """Initialize the stream log writer.

        Args:
            logger: Logger instance to write to.
            level: Log level to use when writing messages.
        """
        self.logger = logger
        self.level = level
        self._buffer = ""

    @property
    def closed(self) -> bool:
        """Check if stream is closed.

        Returns:
            Always False for compatibility with io.IOBase interface.
        """
        return False

    def write(self, message: str) -> None:
        """Write message to the logger.

        Args:
            message: Message to write to the logger.
        """
        if not isinstance(message, str):
            message = str(message)

        if not message.endswith("\n"):
            self._buffer += message
        else:
            self._buffer += message
            self._flush_buffer()

    def flush(self) -> None:
        """Flush any buffered content to the logger."""
        if self._buffer:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        """Internal method to flush buffer content."""
        if self._buffer:
            clean_message = remove_ansi_escape_codes(self._buffer.rstrip())
            self.logger.log(self.level, clean_message)
            self._buffer = ""

    def isatty(self) -> bool:
        """Check if connected to a TTY device.

        Returns:
            Always False for compatibility.
        """
        return False


class RedirectStdHandler(StreamHandler):
    """Handler that dynamically redirects to current sys.stdout/stderr.

    Unlike the standard StreamHandler, this handler always uses the current
    value of sys.stdout/stderr rather than the value at construction time.
    """

    def __init__(self, stream_name: str) -> None:
        """Initialize the redirect handler.

        Args:
            stream_name: Either 'stdout' or 'stderr'.

        Raises:
            ValueError: If stream_name is not 'stdout' or 'stderr'.
        """
        if stream_name not in ("stdout", "stderr"):
            raise ValueError(f"stream_name must be 'stdout' or 'stderr', got '{stream_name}'")

        self._use_stderr = stream_name == "stderr"
        Handler.__init__(self)

    @property
    def stream(self) -> Any:
        """Get the current stream object.

        Returns:
            Current sys.stderr or sys.stdout.
        """
        return sys.stderr if self._use_stderr else sys.stdout


@contextmanager
def redirect_stdout(logger: logging.Logger, level: int = logging.INFO) -> Generator[None, None, None]:
    """Context manager to redirect stdout to a logger.

    Args:
        logger: Logger to redirect stdout to.
        level: Log level to use for messages.

    Yields:
        None

    Example:
        >>> with redirect_stdout(logger, logging.INFO):
        ...     print("This goes to the logger")
    """
    writer = StreamLogWriter(logger, level)
    original_stdout = sys.stdout
    try:
        sys.stdout = writer
        yield
    finally:
        sys.stdout = original_stdout
        writer.flush()


@contextmanager
def redirect_stderr(logger: logging.Logger, level: int = logging.ERROR) -> Generator[None, None, None]:
    """Context manager to redirect stderr to a logger.

    Args:
        logger: Logger to redirect stderr to.
        level: Log level to use for messages.

    Yields:
        None

    Example:
        >>> with redirect_stderr(logger, logging.WARNING):
        ...     print("This goes to the logger", file=sys.stderr)
    """
    writer = StreamLogWriter(logger, level)
    original_stderr = sys.stderr
    try:
        sys.stderr = writer
        yield
    finally:
        sys.stderr = original_stderr
        writer.flush()


def set_logger_context(logger: logging.Logger, context: Any) -> None:
    """Set context for a logger and its handlers.

    Walks the logger hierarchy and attempts to set context on all handlers
    that support it.

    Args:
        logger: Logger to set context for.
        context: Context value to set.
    """
    current_logger: logging.Logger | None = logger

    while current_logger is not None:
        for handler in current_logger.handlers:
            if hasattr(handler, "set_context"):
                try:
                    handler.set_context(context)
                except Exception as e:
                    # Log the error but don't fail the context setting
                    logger.debug(f"Failed to set context on handler {handler}: {e}")

        # Move up the logger hierarchy if propagation is enabled
        current_logger = current_logger.parent if current_logger.propagate else None


# Modern utility functions for common logging patterns


def setup_logger(
    name: str,
    level: int = logging.INFO,
    format_string: str | None = None,
    handler: Handler | None = None,
) -> logging.Logger:
    """Set up a logger with common configuration.

    Args:
        name: Logger name.
        level: Logging level.
        format_string: Custom format string for log messages.
        handler: Custom handler to use.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        if handler is None:
            handler = logging.StreamHandler()

        if format_string is None:
            format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = logging.Formatter(format_string)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


class StructuredLogger(LoggingMixin):
    """Logger with structured logging support.

    Provides methods for logging with structured data that can be easily
    parsed and analyzed.

    Example:
        >>> logger = StructuredLogger()
        >>> logger.log_event("user_login", user_id=123, ip="192.168.1.1")
    """

    def log_event(self, event: str, level: int = logging.INFO, **kwargs: Any) -> None:
        """Log a structured event.

        Args:
            event: Event name/type.
            level: Log level.
            **kwargs: Additional event data.
        """
        event_data = {"event": event, **kwargs}
        self.log.log(level, f"Event: {event}", extra={"event_data": event_data})

    def log_metric(self, metric_name: str, value: int | float, **tags: Any) -> None:
        """Log a metric value.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            **tags: Additional metric tags.
        """
        metric_data = {"metric": metric_name, "value": value, "tags": tags}
        self.log.info(f"Metric: {metric_name}={value}", extra={"metric_data": metric_data})
