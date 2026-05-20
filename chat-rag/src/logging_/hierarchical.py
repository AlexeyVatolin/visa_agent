"""Hierarchical (tree-shaped) console logger.

Use `trace("name", **extra)` as a context manager around each LangGraph node
and non-trivial function. Logs emitted inside the block are indented one level
deeper, producing a console tree alongside LangSmith traces.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_DEFAULT_LOGGER_NAME = "chat_rag"

_INDENT = "   "
_BRANCH = "|- "

_depth: ContextVar[int] = ContextVar("trace_depth", default=0)


def make_prefix(depth: int) -> str:
    if depth <= 0:
        return ""
    return _INDENT * (depth - 1) + _BRANCH


class HierarchicalFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        depth = getattr(record, "trace_depth", _depth.get())
        prefix = make_prefix(depth)
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        return f"[{ts}] {record.levelname:<7} {prefix}{record.getMessage()}"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or _DEFAULT_LOGGER_NAME)


def configure_logging(level: int = logging.INFO) -> None:
    """Wire the hierarchical formatter to the root logger once. Idempotent."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(HierarchicalFormatter())
    root.addHandler(handler)
    root.setLevel(level)


@contextmanager
def trace(name: str, **extra: object) -> Iterator[logging.Logger]:
    """Log node entry and increase indentation depth for nested logs."""
    logger = get_logger()
    extras_str = " ".join(f"{k}={v}" for k, v in extra.items())
    msg = f"{name} {extras_str}".rstrip()
    logger.info(msg)
    token = _depth.set(_depth.get() + 1)
    try:
        yield logger
    finally:
        _depth.reset(token)


def current_depth() -> int:
    return _depth.get()
