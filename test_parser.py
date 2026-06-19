from lexer import LexError
from parser import (
    Arg,
    Binding,
    Call,
    Concat,
    Index,
    Name,
    Num,
    Overlay,
    ParseError,
    Pipe,
    Pipeline,
    Program,
    Sink,
    Slice,
    Speed,
    Stack,
    Str,
    parse,
)

import pytest


def shape(node):
    """Strip positions, reducing the AST to nested tuples for terse equality."""
    if node is None:
        return None
    if isinstance(node, Program):
        return ("Program", [shape(s) for s in node.statements])
    if isinstance(node, Binding):
        return ("Binding", node.name, shape(node.pipeline))
    if isinstance(node, Pipeline):
        return ("Pipeline", shape(node.expr), shape(node.sink))
    if isinstance(node, Sink):
        return ("Sink", shape(node.target))
    if isinstance(node, Pipe):
        return ("Pipe", [shape(x) for x in node.stages])
    if isinstance(node, Overlay):
        return ("Overlay", [shape(x) for x in node.layers])
    if isinstance(node, Concat):
        return ("Concat", [shape(x) for x in node.clips])
    if isinstance(node, Stack):
        return (
            "Stack",
            [shape(x) for x in node.items],
            [(j.axis, j.mode) for j in node.joins],
        )
    if isinstance(node, Call):
        return ("Call", shape(node.func), [(a.name, shape(a.value)) for a in node.args])
    if isinstance(node, Index):
        return ("Index", shape(node.base), [shape(ax) for ax in node.axes])
    if isinstance(node, Slice):
        return ("Slice", shape(node.start), shape(node.stop), shape(node.step))
    if isinstance(node, Speed):
        return ("Speed", shape(node.base), shape(node.factor))
    if isinstance(node, Name):
        return ("Name", node.ident)
    if isinstance(node, Num):
        return ("Num", node.value, node.unit)
    if isinstance(node, Str):
        return ("Str", node.value)
    raise AssertionError(f"unhandled node {node!r}")


def one(src):
    """Parse a single-statement program and return that statement."""
    p = parse(src)
    assert len(p.statements) == 1
    return p.statements[0]


def pexpr(src):
    """The expr of a single bare pipeline (no sink)."""
    st = one(src)
    assert isinstance(st, Pipeline) and st.sink is None
    return st.expr


def test_empty_program():
    assert shape(parse("")) == ("Program", [])


def test_readme_example():
    # im^2 | hflip
    assert shape(pexpr("im^2 | hflip")) == (
        "Pipe",
        [
            ("Speed", ("Name", "im"), ("Num", 2, None)),
            ("Name", "hflip"),
        ],
    )


def test_bare_name():
    assert shape(one("im")) == ("Pipeline", ("Name", "im"), None)


def test_bare_number_keeps_unit():
    assert shape(pexpr("100ms")) == ("Num", 100, "ms")


def test_binding():
    assert shape(one("out = im | blur(2)")) == (
        "Binding",
        "out",
        (
            "Pipeline",
            (
                "Pipe",
                [
                    ("Name", "im"),
                    ("Call", ("Name", "blur"), [(None, ("Num", 2, None))]),
                ],
            ),
            None,
        ),
    )


def test_sink_to_string():
    assert shape(one('im > "out.mp4"')) == (
        "Pipeline",
        ("Name", "im"),
        ("Sink", ("Str", "out.mp4")),
    )


def test_sink_to_ident_pool():
    assert shape(one("im > pool")) == (
        "Pipeline",
        ("Name", "im"),
        ("Sink", ("Name", "pool")),
    )


def test_empty_call():
    assert shape(pexpr("reverse()")) == ("Call", ("Name", "reverse"), [])


def test_positional_and_keyword_args():
    assert shape(pexpr('text("hi", pos=top)')) == (
        "Call",
        ("Name", "text"),
        [(None, ("Str", "hi")), ("pos", ("Name", "top"))],
    )


def test_keyword_arg_expr_value():
    assert shape(pexpr("resize(w=320, h=240)")) == (
        "Call",
        ("Name", "resize"),
        [("w", ("Num", 320, None)), ("h", ("Num", 240, None))],
    )


def test_multi_axis_slice():
    assert shape(pexpr("im[0:10, :, :]")) == (
        "Index",
        ("Name", "im"),
        [
            ("Slice", ("Num", 0, None), ("Num", 10, None), None),
            ("Slice", None, None, None),
            ("Slice", None, None, None),
        ],
    )


@pytest.mark.parametrize(
    "src,expected",
    [
        ("im[:]", ("Slice", None, None, None)),
        ("im[::2]", ("Slice", None, None, ("Num", 2, None))),
        ("im[a:]", ("Slice", ("Name", "a"), None, None)),
        ("im[:b]", ("Slice", None, ("Name", "b"), None)),
        ("im[a:b:c]", ("Slice", ("Name", "a"), ("Name", "b"), ("Name", "c"))),
        ("im[5]", ("Num", 5, None)),  # single index is a bare expr, not a Slice
    ],
)
def test_slice_variants(src, expected):
    idx = pexpr(src)
    assert isinstance(idx, Index) and len(idx.axes) == 1
    assert shape(idx.axes[0]) == expected


@pytest.mark.parametrize(
    "src,factor",
    [
        ("x^2", ("Num", 2, None)),
        ("x^-1", ("Num", -1, None)),
        ("x^var", ("Name", "var")),
        ("x^(a >> b)", ("Concat", [("Name", "a"), ("Name", "b")])),
    ],
)
def test_speed_forms(src, factor):
    assert shape(pexpr(src)) == ("Speed", ("Name", "x"), factor)


def test_concat_with_transition_is_flat():
    assert shape(pexpr("a >> dissolve(1s) >> b")) == (
        "Concat",
        [
            ("Name", "a"),
            ("Call", ("Name", "dissolve"), [(None, ("Num", 1, "s"))]),
            ("Name", "b"),
        ],
    )


def test_mixed_layout_with_modes():
    assert shape(pexpr("a & b /fit c &scale d")) == (
        "Stack",
        [("Name", "a"), ("Name", "b"), ("Name", "c"), ("Name", "d")],
        [("h", None), ("v", "fit"), ("h", "scale")],
    )


def test_overlay_chain_is_flat():
    assert shape(pexpr("a over b over c")) == (
        "Overlay",
        [("Name", "a"), ("Name", "b"), ("Name", "c")],
    )


def test_pipe_binds_tighter_than_concat():
    # a | b >> c  ==  (a | b) >> c
    assert shape(pexpr("a | b >> c")) == (
        "Concat",
        [("Pipe", [("Name", "a"), ("Name", "b")]), ("Name", "c")],
    )


def test_parens_are_transparent():
    assert shape(pexpr("(a | b)")) == ("Pipe", [("Name", "a"), ("Name", "b")])
    assert shape(pexpr("(a) >> b")) == (
        "Concat",
        [("Name", "a"), ("Name", "b")],
    )


def test_postfix_chaining():
    assert shape(pexpr("f(x)[0]^2")) == (
        "Speed",
        ("Index", ("Call", ("Name", "f"), [(None, ("Name", "x"))]), [("Num", 0, None)]),
        ("Num", 2, None),
    )


def test_multi_statement_program_positions():
    p = parse("a = im\nb = a^2")
    assert len(p.statements) == 2
    assert isinstance(p.statements[0], Binding)
    assert isinstance(p.statements[1], Binding)
    # second statement begins on line 2
    assert p.statements[1].line == 2


@pytest.mark.parametrize(
    "src",
    [
        "im[]",  # index requires at least one axis
        "a > b c",  # trailing tokens after a complete statement
        "a b",  # two primaries, no operator
        'a > "out" | b',  # sink is terminal; cannot pipe after it
        "im > 5",  # sink target must be string or ident
        "(a",  # unterminated paren
        "im[0",  # unterminated bracket
        'im^"x"',  # signed_atom excludes strings
        "| a",  # leading operator
        "out =",  # binding with no RHS
    ],
)
def test_parse_errors(src):
    with pytest.raises(ParseError):
        parse(src)


def test_lex_error_propagates():
    with pytest.raises(LexError):
        parse('"oops')
