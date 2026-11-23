"""Helpers for preserving useful exception diagnostics."""

from __future__ import annotations


def get_diagnostic_exception_message(error: BaseException) -> str:
    """Return an exception message, falling back to its type name."""
    return str(error).strip() or type(error).__name__
