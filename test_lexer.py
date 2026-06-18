from lexer import LexError, Number, TokenType, tokenize

import pytest


def types(source):
    return [t.type for t in tokenize(source)]


def kinds(source):
    """(type, value) pairs excluding the trailing EOF."""
    return [(t.type, t.value) for t in tokenize(source)[:-1]]


def test_ends_with_eof():
    toks = tokenize("")
    assert [t.type for t in toks] == [TokenType.EOF]


def test_readme_example():
    # im^2 | hflip
    assert kinds("im^2 | hflip") == [
        (TokenType.IDENT, "im"),
        (TokenType.SPEED, None),
        (TokenType.NUMBER, Number(2)),
        (TokenType.PIPE, None),
        (TokenType.IDENT, "hflip"),
    ]


def test_over_keyword_vs_ident():
    assert kinds("over") == [(TokenType.OVER, None)]
    assert kinds("overflow") == [(TokenType.IDENT, "overflow")]


def test_concat_vs_sink_maximal_munch():
    assert types(">>") == [TokenType.CONCAT, TokenType.EOF]
    assert types(">") == [TokenType.SINK, TokenType.EOF]
    assert types(">>>") == [TokenType.CONCAT, TokenType.SINK, TokenType.EOF]


def test_numbers_units_and_floats():
    assert kinds("100ms") == [(TokenType.NUMBER, Number(100, "ms"))]
    assert kinds("5s") == [(TokenType.NUMBER, Number(5, "s"))]
    assert kinds("30f") == [(TokenType.NUMBER, Number(30, "f"))]
    assert kinds("50%") == [(TokenType.NUMBER, Number(50, "%"))]
    assert kinds("2.5") == [(TokenType.NUMBER, Number(2.5))]
    assert kinds("2.5s") == [(TokenType.NUMBER, Number(2.5, "s"))]


def test_negative_number_has_no_minus_operator():
    # ^-1 is SPEED then NUMBER(-1)
    assert kinds("^-1") == [(TokenType.SPEED, None), (TokenType.NUMBER, Number(-1))]


def test_unit_must_not_run_into_identifier():
    assert kinds("5second") == [
        (TokenType.NUMBER, Number(5)),
        (TokenType.IDENT, "second"),
    ]


def test_hcat_mode_adjacency():
    # &fit -> mode attaches; & fit -> HCAT then IDENT
    assert kinds("a&fit b") == [
        (TokenType.IDENT, "a"),
        (TokenType.HCAT, "fit"),
        (TokenType.IDENT, "b"),
    ]
    assert kinds("a& fit b") == [
        (TokenType.IDENT, "a"),
        (TokenType.HCAT, None),
        (TokenType.IDENT, "fit"),
        (TokenType.IDENT, "b"),
    ]
    assert kinds("a/scale b") == [
        (TokenType.IDENT, "a"),
        (TokenType.VCAT, "scale"),
        (TokenType.IDENT, "b"),
    ]


def test_strings_decode_escapes():
    assert kinds(r'"hi\nthere"') == [(TokenType.STRING, "hi\nthere")]
    assert kinds(r'"a\"b"') == [(TokenType.STRING, 'a"b')]


def test_unterminated_string_raises():
    with pytest.raises(LexError):
        tokenize('"oops')


def test_comments_are_skipped():
    assert kinds("im # a comment") == [(TokenType.IDENT, "im")]
    assert types("# whole line") == [TokenType.EOF]


def test_separators_collapse_and_trim():
    # leading/trailing/consecutive separators reduce to single SEPs
    assert types("\n\na;;b\n\n") == [
        TokenType.IDENT,
        TokenType.SEP,
        TokenType.IDENT,
        TokenType.EOF,
    ]


def test_newlines_inside_brackets_are_not_separators():
    assert types("(a\nb)") == [
        TokenType.LPAREN,
        TokenType.IDENT,
        TokenType.IDENT,
        TokenType.RPAREN,
        TokenType.EOF,
    ]
    # ...but outside brackets they separate
    assert types("a\nb") == [
        TokenType.IDENT,
        TokenType.SEP,
        TokenType.IDENT,
        TokenType.EOF,
    ]


def test_slice_syntax():
    assert types("im[0:10, :, :]") == [
        TokenType.IDENT,
        TokenType.LBRACK,
        TokenType.NUMBER,
        TokenType.COLON,
        TokenType.NUMBER,
        TokenType.COMMA,
        TokenType.COLON,
        TokenType.COMMA,
        TokenType.COLON,
        TokenType.RBRACK,
        TokenType.EOF,
    ]


def test_position_tracking():
    toks = tokenize("ab\n  cd")
    ab, sep, cd, _eof = toks
    assert (ab.line, ab.col) == (1, 1)
    assert (cd.line, cd.col) == (2, 3)


def test_unexpected_character():
    with pytest.raises(LexError):
        tokenize("@")
