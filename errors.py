from __future__ import annotations

from typing import Optional


class HeddleError(Exception):
    """A runtime error raised while interpreting a program.

    Mirrors ParseError / LexError: it carries an optional source position so the
    CLI can report `line:col: message`. The position is omitted for errors that
    aren't tied to a specific token (e.g. a missing source file on disk).
    """

    def __init__(
        self, message: str, line: Optional[int] = None, col: Optional[int] = None
    ):
        if line is not None and col is not None:
            super().__init__(f"{line}:{col}: {message}")
        else:
            super().__init__(message)
        self.message = message
        self.line = line
        self.col = col
