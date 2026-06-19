from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from lexer import Token, TokenType, tokenize

# NOTE: Associative operator chains (|, over, &, /, >>) are represented as flat n-ary nodes
# rather than nested binary ones for ease of interpretation, e.g., a pipe is a list of stages,
# a timeline is a list of clips/transitions. Postfix operators (call/index/speed) are nested.


## expressions


@dataclass
class Name:
    ident: str
    line: int
    col: int


@dataclass
class Num:
    value: Union[int, float]
    unit: Optional[str]  # None | "s" | "ms" | "f" | "%"
    line: int
    col: int


@dataclass
class Str:
    value: str
    line: int
    col: int


@dataclass
class Pipe:
    stages: list[Expr]  # >= 2 operands: a | b | c
    line: int
    col: int


@dataclass
class Overlay:
    layers: list[Expr]  # >= 2 operands: a over b over c
    line: int
    col: int


@dataclass
class Join:
    axis: str  # "h" (HCAT &) or "v" (VCAT /)
    mode: Optional[str]
    line: int
    col: int


@dataclass
class Stack:
    items: list[Expr]  # n operands
    joins: list[Join]  # joins[i] connects items[i] and items[i+1]
    line: int
    col: int


@dataclass
class Concat:
    clips: list[Expr]  # >= 2 operands: a >> b >> c (clips/transitions)
    line: int
    col: int


## postfix


@dataclass
class Arg:
    name: Optional[str]  # None => positional; str => keyword
    value: Expr
    line: int
    col: int


@dataclass
class Call:
    func: Expr
    args: list[Arg]
    line: int
    col: int


@dataclass
class Slice:
    start: Optional[Expr]
    stop: Optional[Expr]
    step: Optional[Expr]
    line: int
    col: int


@dataclass
class Index:
    base: Expr
    axes: list[Slice | Expr]  # Slice (range) or bare expr (single index); len >= 1
    line: int
    col: int


@dataclass
class Speed:
    base: Expr
    factor: Expr  # signed_atom: Num | Name | parenthesized expr
    line: int
    col: int


## statements


@dataclass
class Sink:
    target: Str | Name  # Str: file (format from ext); Name: pool
    line: int
    col: int


@dataclass
class Pipeline:
    expr: Expr
    sink: Optional[Sink]
    line: int
    col: int


@dataclass
class Binding:
    name: str
    pipeline: Pipeline
    line: int
    col: int


@dataclass
class Program:
    statements: list[Binding | Pipeline]
    line: int
    col: int


# An expression node: anything that evaluates to a media value. Slice, Arg,
# Join, and the statement nodes are excluded — they only appear in fixed spots.
Expr = Union[Name, Num, Str, Pipe, Overlay, Stack, Concat, Call, Index, Speed]


class ParseError(Exception):
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"{line}:{col}: {message}")
        self.message = message
        self.line = line
        self.col = col


# Tokens that bound an axis part; an optional start/stop/step is absent exactly
# when the cursor sits on one of these where that part would begin.
_AXIS_END = (TokenType.COLON, TokenType.COMMA, TokenType.RBRACK)


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.i = 0

    ## cursor helpers

    def _peek(self, offset: int = 0) -> Token:
        j = self.i + offset
        # Clamp to the trailing EOF so lookahead past the end never raises.
        return self.tokens[j] if j < len(self.tokens) else self.tokens[-1]

    def _at(self, *types: TokenType) -> bool:
        return self._peek().type in types

    def _at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _advance(self) -> Token:
        tok = self.tokens[self.i]
        if tok.type != TokenType.EOF:
            self.i += 1
        return tok

    def _expect(self, ttype: TokenType, what: str) -> Token:
        tok = self._peek()
        if tok.type != ttype:
            raise self._error(f"expected {what}", tok)
        return self._advance()

    def _error(self, message: str, tok: Optional[Token] = None) -> ParseError:
        tok = tok or self._peek()
        return ParseError(message, tok.line, tok.col)

    ## program / statements

    def parse(self) -> Program:
        # program = [ statement ] , { SEP , [ statement ] }
        first = self._peek()
        statements: list = []
        if not self._at_end():
            statements.append(self.parse_statement())
            while self._at(TokenType.SEP):
                self._advance()
                statements.append(self.parse_statement())
        self._expect(TokenType.EOF, "end of input")
        return Program(statements, first.line, first.col)

    def parse_statement(self):
        # statement = binding | pipeline ; binding = IDENT ASSIGN pipeline
        if self._at(TokenType.IDENT) and self._peek(1).type == TokenType.ASSIGN:
            name = self._advance()  # IDENT
            self._advance()  # ASSIGN
            return Binding(name.value, self.parse_pipeline(), name.line, name.col)
        return self.parse_pipeline()

    def parse_pipeline(self):
        # pipeline = expr , [ SINK , target ]
        expr = self.parse_expr()
        sink = None
        if self._at(TokenType.SINK):
            gt = self._advance()
            sink = Sink(self.parse_target(), gt.line, gt.col)
        return Pipeline(expr, sink, expr.line, expr.col)

    def parse_target(self):
        # target = STRING | IDENT
        tok = self._peek()
        if tok.type == TokenType.STRING:
            self._advance()
            return Str(tok.value, tok.line, tok.col)
        if tok.type == TokenType.IDENT:
            self._advance()
            return Name(tok.value, tok.line, tok.col)
        raise self._error("expected sink target (string or identifier)", tok)

    ## expression chains (lowest to highest precedence)

    def parse_expr(self):
        return self.parse_sequence()

    def parse_sequence(self):
        # sequence = layout , { CONCAT , layout }
        first = self.parse_layout()
        clips = [first]
        while self._at(TokenType.CONCAT):
            self._advance()
            clips.append(self.parse_layout())
        if len(clips) == 1:
            return first
        return Concat(clips, first.line, first.col)

    def parse_layout(self):
        # layout = overlay , { ( HCAT | VCAT ) , overlay }
        first = self.parse_overlay()
        items = [first]
        joins: list = []
        while self._at(TokenType.HCAT, TokenType.VCAT):
            op = self._advance()
            axis = "h" if op.type == TokenType.HCAT else "v"
            joins.append(Join(axis, op.value, op.line, op.col))
            items.append(self.parse_overlay())
        if len(items) == 1:
            return first
        return Stack(items, joins, first.line, first.col)

    def parse_overlay(self):
        # overlay = pipe , { OVER , pipe }
        first = self.parse_pipe()
        layers = [first]
        while self._at(TokenType.OVER):
            self._advance()
            layers.append(self.parse_pipe())
        if len(layers) == 1:
            return first
        return Overlay(layers, first.line, first.col)

    def parse_pipe(self):
        # pipe = postfix , { PIPE , postfix }
        first = self.parse_postfix()
        stages = [first]
        while self._at(TokenType.PIPE):
            self._advance()
            stages.append(self.parse_postfix())
        if len(stages) == 1:
            return first
        return Pipe(stages, first.line, first.col)

    ## postfix

    def parse_postfix(self):
        # postfix = primary , { call | index | speed }
        node = self.parse_primary()
        while True:
            tok = self._peek()
            if tok.type == TokenType.LPAREN:
                node = self.parse_call(node)
            elif tok.type == TokenType.LBRACK:
                node = self.parse_index(node)
            elif tok.type == TokenType.SPEED:
                self._advance()  # "^"
                factor = self.parse_signed_atom()
                node = Speed(node, factor, node.line, node.col)
            else:
                return node

    def parse_call(self, func):
        # call = LPAREN , [ arg_list ] , RPAREN
        self._expect(TokenType.LPAREN, "'('")
        args: list = []
        if not self._at(TokenType.RPAREN):
            args.append(self.parse_arg())
            while self._at(TokenType.COMMA):
                self._advance()
                args.append(self.parse_arg())
        self._expect(TokenType.RPAREN, "')'")
        return Call(func, args, func.line, func.col)

    def parse_arg(self):
        # arg = IDENT , ASSIGN , expr | expr
        if self._at(TokenType.IDENT) and self._peek(1).type == TokenType.ASSIGN:
            name = self._advance()  # IDENT
            self._advance()  # ASSIGN
            return Arg(name.value, self.parse_expr(), name.line, name.col)
        value = self.parse_expr()
        return Arg(None, value, value.line, value.col)

    def parse_index(self, base):
        # index = LBRACK , axis , { COMMA , axis } , RBRACK
        self._expect(TokenType.LBRACK, "'['")
        axes = [self.parse_axis()]
        while self._at(TokenType.COMMA):
            self._advance()
            axes.append(self.parse_axis())
        self._expect(TokenType.RBRACK, "']'")
        return Index(base, axes, base.line, base.col)

    def parse_axis(self):
        # axis = slice | expr ; slice = [expr] COLON [expr] [COLON [expr]]
        pos = self._peek()
        if self._at(TokenType.COLON):
            return self._parse_slice_from(None, pos)
        first = self.parse_expr()
        if self._at(TokenType.COLON):
            return self._parse_slice_from(first, pos)
        return first  # single-index axis: a bare expr, not a Slice

    def _parse_slice_from(self, start, pos):
        # positioned at the first ':'
        self._advance()  # COLON
        stop = None
        if not self._at(*_AXIS_END):
            stop = self.parse_expr()
        step = None
        if self._at(TokenType.COLON):
            self._advance()
            if not self._at(TokenType.COMMA, TokenType.RBRACK):
                step = self.parse_expr()
        return Slice(start, stop, step, pos.line, pos.col)

    def parse_signed_atom(self):
        # signed_atom = NUMBER | IDENT | LPAREN expr RPAREN
        tok = self._peek()
        if tok.type == TokenType.NUMBER:
            self._advance()
            return Num(tok.value.value, tok.value.unit, tok.line, tok.col)
        if tok.type == TokenType.IDENT:
            self._advance()
            return Name(tok.value, tok.line, tok.col)
        if tok.type == TokenType.LPAREN:
            self._advance()
            inner = self.parse_expr()
            self._expect(TokenType.RPAREN, "')'")
            return inner  # parens transparent
        raise self._error("expected number, identifier, or '(' after '^'", tok)

    def parse_primary(self):
        # primary = IDENT | NUMBER | STRING | LPAREN expr RPAREN
        tok = self._peek()
        if tok.type == TokenType.IDENT:
            self._advance()
            return Name(tok.value, tok.line, tok.col)
        if tok.type == TokenType.NUMBER:
            self._advance()
            return Num(tok.value.value, tok.value.unit, tok.line, tok.col)
        if tok.type == TokenType.STRING:
            self._advance()
            return Str(tok.value, tok.line, tok.col)
        if tok.type == TokenType.LPAREN:
            self._advance()
            inner = self.parse_expr()
            self._expect(TokenType.RPAREN, "')'")
            return inner  # parens transparent: no node
        raise self._error("expected expression", tok)


def parse(source: str) -> Program:
    return Parser(tokenize(source)).parse()
