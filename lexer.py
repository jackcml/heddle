from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NamedTuple, Optional, Union


class TokenType(Enum):
    IDENT = auto()
    STRING = auto()
    NUMBER = auto()
    PIPE = auto()  # |
    CONCAT = auto()  # >>
    SINK = auto()  # >
    SPEED = auto()  # ^
    OVER = auto()  # over
    HCAT = auto()  # & [mode]
    VCAT = auto()  # / [mode]
    ASSIGN = auto()  # =
    LPAREN = auto()  # (
    RPAREN = auto()  # )
    LBRACK = auto()  # [
    RBRACK = auto()  # ]
    COMMA = auto()  # ,
    COLON = auto()  # :
    SEP = auto()  # ; or newline
    EOF = auto()


class Number(NamedTuple):
    value: Union[int, float]
    unit: Optional[str] = None

    def __repr__(self) -> str:
        return f"{self.value}{self.unit or ''}"


@dataclass
class Token:
    type: TokenType

    #  IDENT    => str (name)
    #  NUMBER   => Number
    #  {H/V}CAT => str or None (mode)
    #  _  => None
    value: object = None

    line: int = 1
    col: int = 1

    def __str__(self) -> str:
        if self.value is None:
            return self.type.name
        return f"{self.type.name}({self.value!r})"


class LexError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col


_SINGLE = {
    "|": TokenType.PIPE,
    "^": TokenType.SPEED,
    "=": TokenType.ASSIGN,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "[": TokenType.LBRACK,
    "]": TokenType.RBRACK,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
}
_KEYWORDS = {"over": TokenType.OVER}
_UNITS = ("ms", "s", "f", "%")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "0": "\0"}


class Lexer:
    def __init__(self, src: str):
        self.src = src
        self.i = 0
        self.line = 1
        self.col = 1
        self.depth = 0  # nesting depth of ()/[] to suppress newline SEPs
        self.tokens: list[Token] = []

    ## character helpers

    def _peek(self, offset: int = 0) -> str:
        j = self.i + offset
        return self.src[j] if j < len(self.src) else ""

    def _advance(self) -> str:
        ch = self.src[self.i]
        self.i += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _at_end(self) -> bool:
        return self.i >= len(self.src)

    ## main loop

    def tokenize(self) -> list[Token]:
        while not self._at_end():
            ch = self._peek()
            if ch in " \t\r":
                self._advance()
            elif ch == "\n":
                line, col = self.line, self.col
                self._advance()
                if self.depth == 0:
                    self._emit_sep(line, col)
            elif ch == ";":
                line, col = self.line, self.col
                self._advance()
                self._emit_sep(line, col)
            elif ch == "#":
                self._comment()
            elif ch == '"':
                self._string()
            elif ch.isdigit() or (ch == "-" and self._peek(1).isdigit()):
                self._number()
            elif ch.isalpha():
                self._ident()
            elif ch == "&":
                self._cat(TokenType.HCAT)
            elif ch == "/":
                self._cat(TokenType.VCAT)
            elif ch == ">":
                self._gt()
            elif ch in _SINGLE:
                self._single(ch)
            else:
                raise LexError(f"unexpected character {ch!r}", self.line, self.col)

        # trim trailing SEP to prevent empty statement; finalize with EOF
        if self.tokens and self.tokens[-1].type == TokenType.SEP:
            self.tokens.pop()
        self.tokens.append(Token(TokenType.EOF, None, self.line, self.col))
        return self.tokens

    ## scanners

    def _emit_sep(self, line: int, col: int) -> None:
        # drop leading and collapse adjacent SEPs
        if self.tokens and self.tokens[-1].type != TokenType.SEP:
            self.tokens.append(Token(TokenType.SEP, None, line, col))

    def _comment(self) -> None:
        while not self._at_end() and self._peek() != "\n":
            self._advance()

    def _single(self, ch: str) -> None:
        line, col = self.line, self.col
        ttype = _SINGLE[ch]
        self._advance()
        
        if ttype in (TokenType.LPAREN, TokenType.LBRACK):
            self.depth += 1
        elif ttype in (TokenType.RPAREN, TokenType.RBRACK):
            self.depth = max(0, self.depth - 1)

        self.tokens.append(Token(ttype, None, line, col))

    def _gt(self) -> None:
        line, col = self.line, self.col
        self._advance()

        if self._peek() == ">":
            self._advance()
            self.tokens.append(Token(TokenType.CONCAT, None, line, col))
        else:
            self.tokens.append(Token(TokenType.SINK, None, line, col))

    def _string(self) -> None:
        line, col = self.line, self.col
        self._advance()  # opening quote
        chars: list[str] = []

        while True:
            if self._at_end():
                raise LexError("unterminated string", line, col)
            ch = self._peek()
            if ch == '"':
                self._advance()
                break
            if ch == "\\":
                self._advance()
                if self._at_end():
                    raise LexError("unterminated string", line, col)
                esc = self._advance()
                chars.append(_ESCAPES.get(esc, esc))
            else:
                chars.append(self._advance())

        self.tokens.append(Token(TokenType.STRING, "".join(chars), line, col))

    def _number(self) -> None:
        line, col = self.line, self.col
        start = self.i

        if self._peek() == "-":
            self._advance()

        while self._peek().isdigit():
            self._advance()

        is_float = False
        if self._peek() == "." and self._peek(1).isdigit():
            is_float = True
            self._advance()  # "."
            while self._peek().isdigit():
                self._advance()

        text = self.src[start : self.i]
        value: Union[int, float] = float(text) if is_float else int(text)
        self.tokens.append(
            Token(TokenType.NUMBER, Number(value, self._unit()), line, col)
        )

    def _unit(self) -> Optional[str]:
        for u in _UNITS:
            if self.src.startswith(u, self.i):
                after = self.src[self.i + len(u) : self.i + len(u) + 1]
                # differentiate identifier from unit (`5sab` is `[5][sab]`, not `[5s][ab]`)
                if u != "%" and (after.isalnum() or after == "_"):
                    continue
                for _ in range(len(u)):
                    self._advance()
                return u
        return None

    def _ident(self) -> None:
        line, col = self.line, self.col
        start = self.i
        self._advance()

        while self._peek().isalnum() or self._peek() == "_":
            self._advance()
        name = self.src[start : self.i]

        ttype = _KEYWORDS.get(name)
        if ttype is not None:
            self.tokens.append(Token(ttype, None, line, col))
        else:
            self.tokens.append(Token(TokenType.IDENT, name, line, col))

    def _cat(self, ttype: TokenType) -> None:
        line, col = self.line, self.col
        self._advance() # `&` or `/`
        mode: Optional[str] = None

        # mode only attaches without whitespace in between
        if self._peek().isalpha():
            start = self.i
            while self._peek().isalpha():
                self._advance()
            mode = self.src[start : self.i]
        self.tokens.append(Token(ttype, mode, line, col))


def tokenize(source: str) -> list[Token]:
    return Lexer(source).tokenize()
