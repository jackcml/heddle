from PIL import Image

import pytest

import interpreter
from clip import Clip, load, save
from errors import HeddleError
from interpreter import Env, eval_node, run_program
from parser import parse
from registry import Param, Transform, lookup, transform
from transforms import apply_speed

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def solid(color, size=(2, 2)):
    return Image.new("RGBA", size, color)


def make_clip(n=1, color=(255, 0, 0, 255), dur=100, size=(2, 2)):
    return Clip([solid(color, size) for _ in range(n)], [dur] * n, 0)


def expr_of(src):
    """The expression node of a single bare (sinkless) pipeline statement."""
    return parse(src).statements[0].expr


# ---------------------------------------------------------------------------
# Clip invariants
# ---------------------------------------------------------------------------


def test_clip_frames_and_durations_parallel():
    clip = make_clip(n=3)
    assert len(clip.frames) == len(clip.durations) == 3
    assert clip.is_animated
    assert not make_clip(n=1).is_animated


# ---------------------------------------------------------------------------
# transforms: hflip
# ---------------------------------------------------------------------------


def test_hflip_swaps_pixels():
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (255, 0, 0, 255))  # left red
    img.putpixel((1, 0), (0, 255, 0, 255))  # right green
    clip = Clip([img], [100], 0)

    out = lookup("hflip").func(clip)

    assert out.frames[0].getpixel((0, 0)) == (0, 255, 0, 255)  # left now green
    assert out.frames[0].getpixel((1, 0)) == (255, 0, 0, 255)  # right now red


def test_hflip_keeps_frame_count_and_durations():
    clip = make_clip(n=4, dur=70)
    out = lookup("hflip").func(clip)
    assert len(out.frames) == 4
    assert out.durations == [70, 70, 70, 70]


# ---------------------------------------------------------------------------
# transforms: apply_speed (duration scaling)
# ---------------------------------------------------------------------------


def test_apply_speed_speeds_up():
    out = apply_speed(make_clip(n=2, dur=100), 2)
    assert out.durations == [50, 50]
    assert len(out.frames) == 2  # frame count unchanged


def test_apply_speed_slows_down():
    out = apply_speed(make_clip(n=2, dur=100), 0.5)
    assert out.durations == [200, 200]


def test_apply_speed_negative_reverses():
    clip = Clip(
        [solid((1, 0, 0, 255)), solid((2, 0, 0, 255))],
        [100, 200],
        0,
    )
    out = apply_speed(clip, -1)
    # frames and durations both reversed; |k| == 1 so timing magnitude is kept
    assert out.durations == [200, 100]
    assert out.frames[0].getpixel((0, 0)) == (2, 0, 0, 255)


def test_apply_speed_zero_raises():
    with pytest.raises(HeddleError):
        apply_speed(make_clip(), 0)


def test_apply_speed_floors_at_one_ms():
    out = apply_speed(make_clip(n=1, dur=1), 10)
    assert out.durations == [1]


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_lookup_known_and_unknown():
    assert isinstance(lookup("hflip"), Transform)
    assert lookup("does_not_exist") is None


def test_decorator_registers(monkeypatch):
    @transform("__tmp_noop__")
    def noop(clip):
        return clip

    try:
        assert lookup("__tmp_noop__").func is noop
    finally:
        import registry

        registry._REGISTRY.pop("__tmp_noop__", None)


def test_param_required_vs_default():
    assert Param("x").required
    assert not Param("x", 5).required


# ---------------------------------------------------------------------------
# argument binding (through a throwaway 1-param transform)
# ---------------------------------------------------------------------------


def _with_tmp_transform(name, params, fn, body):
    transform(name, params=params)(fn)
    try:
        body()
    finally:
        import registry

        registry._REGISTRY.pop(name, None)


def test_arg_binding_positional_and_keyword():
    captured = {}

    def grab(clip, amount):
        captured["amount"] = amount
        return clip

    def body():
        env = Env()
        eval_node(expr_of("tmpamt(3)"), make_clip(), env)
        assert captured["amount"] == 3
        eval_node(expr_of("tmpamt(amount=4)"), make_clip(), env)
        assert captured["amount"] == 4

    _with_tmp_transform("tmpamt", ("amount",), grab, body)


def test_arg_binding_missing_required_raises():
    def body():
        with pytest.raises(HeddleError):
            eval_node(expr_of("tmpreq()"), make_clip(), Env())

    _with_tmp_transform("tmpreq", ("amount",), lambda clip, amount: clip, body)


def test_arg_binding_unknown_keyword_raises():
    def body():
        with pytest.raises(HeddleError):
            eval_node(expr_of("tmpkw(nope=1)"), make_clip(), Env())

    _with_tmp_transform("tmpkw", ("amount",), lambda clip, amount: clip, body)


def test_arg_binding_too_many_positional_raises():
    def body():
        with pytest.raises(HeddleError):
            eval_node(expr_of("tmpone(1, 2)"), make_clip(), Env())

    _with_tmp_transform("tmpone", ("amount",), lambda clip, amount: clip, body)


# ---------------------------------------------------------------------------
# eval dispatch (no disk IO: load/resolve_source monkeypatched)
# ---------------------------------------------------------------------------


def test_eval_pipeline_speed_then_flip(monkeypatch):
    src_clip = Clip(
        [
            Image.new("RGBA", (2, 1), (255, 0, 0, 255)),
        ],
        [100],
        0,
    )
    src_clip.frames[0].putpixel((1, 0), (0, 255, 0, 255))  # left red, right green

    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: src_clip)

    out = eval_node(expr_of("im^2 | hflip"), None, Env())

    assert out.durations == [50]  # ^2 halved the duration
    assert out.frames[0].getpixel((0, 0)) == (0, 255, 0, 255)  # hflip swapped sides


def test_binding_resolves_downstream(monkeypatch):
    src_clip = make_clip()
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: src_clip)

    env = run_program(parse("base = im"), cwd=".")
    assert env.names["base"] is src_clip


# ---------------------------------------------------------------------------
# speed-factor unit validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src,expected",
    [("im^2", [50]), ("im^200%", [50]), ("im^50%", [200])],
)
def test_speed_factor_units_ok(monkeypatch, src, expected):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip(n=1, dur=100))
    out = eval_node(expr_of(src), None, Env())
    assert out.durations == expected


@pytest.mark.parametrize("src", ["im^2s", "im^100ms", "im^3f"])
def test_speed_factor_bad_unit_raises(monkeypatch, src):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip())
    with pytest.raises(HeddleError):
        eval_node(expr_of(src), None, Env())


# ---------------------------------------------------------------------------
# extension seams + sink rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "src",
    ["a over b", "a & b", "a / b", "a >> b", "a[0]"],
)
def test_unimplemented_operators_raise(monkeypatch, src):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "x.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip())
    with pytest.raises(NotImplementedError):
        eval_node(expr_of(src), None, Env())


def test_no_sink_pipeline_raises(monkeypatch):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip())
    with pytest.raises(HeddleError):
        run_program(parse("im | hflip"), cwd=".")


def test_unknown_transform_raises(monkeypatch):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip())
    with pytest.raises(HeddleError):
        eval_node(expr_of("im | bogus"), None, Env())


def test_scalar_as_media_raises():
    with pytest.raises(HeddleError):
        eval_node(expr_of("5"), None, Env())


# ---------------------------------------------------------------------------
# disk round-trip
# ---------------------------------------------------------------------------


def test_gif_roundtrip(tmp_path):
    clip = Clip(
        [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))],
        [120, 180],
        0,
    )
    path = tmp_path / "out.gif"
    save(clip, str(path))
    assert path.exists()

    reloaded = load(str(path))
    assert len(reloaded.frames) == 2
    assert reloaded.durations == [120, 180]
