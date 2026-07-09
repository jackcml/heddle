from PIL import Image

import pytest

import interpreter
from clip import Clip, load, save
from errors import HeddleError
from interpreter import Env, eval_node, run_program
from parser import parse
from registry import Param, Transform, lookup, transform
from transforms import Dissolve, apply_concat, apply_overlay, apply_speed, apply_stack

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
# transforms: blur
# ---------------------------------------------------------------------------


def test_blur_spreads_pixels_and_preserves_clip_metadata():
    frame = Image.new("RGBA", (9, 9), (0, 0, 0, 255))
    frame.putpixel((4, 4), (255, 255, 255, 255))
    clip = Clip([frame, frame.copy()], [70, 90], 3)

    out = lookup("blur").func(clip, 1)

    assert 0 < out.frames[0].getpixel((4, 4))[0] < 255
    assert out.frames[0].getpixel((4, 3))[0] > 0
    assert out.durations == [70, 90]
    assert out.loop == 3


def test_blur_rejects_negative_stdev():
    with pytest.raises(HeddleError, match="non-negative"):
        lookup("blur").func(make_clip(), -1)


# ---------------------------------------------------------------------------
# transforms: scale
# ---------------------------------------------------------------------------


def test_scale_resizes_each_frame_uniformly_and_preserves_metadata():
    clip = make_clip(n=2, dur=75, size=(4, 6))
    clip.loop = 2

    out = lookup("scale").func(clip, 1.5)

    assert [frame.size for frame in out.frames] == [(6, 9), (6, 9)]
    assert out.durations == [75, 75]
    assert out.loop == 2


@pytest.mark.parametrize("factor", [0, -1, float("inf"), "large"])
def test_scale_rejects_invalid_factor(factor):
    with pytest.raises(HeddleError, match="positive finite"):
        lookup("scale").func(make_clip(), factor)


# ---------------------------------------------------------------------------
# transforms: resize
# ---------------------------------------------------------------------------


def test_resize_uses_exact_dimensions_and_preserves_metadata():
    clip = make_clip(n=2, dur=80, size=(7, 4))
    clip.loop = 5

    out = lookup("resize").func(clip, 3, 6)

    assert [frame.size for frame in out.frames] == [(3, 6), (3, 6)]
    assert out.durations == [80, 80]
    assert out.loop == 5


@pytest.mark.parametrize("dimensions", [(0, 2), (2, -1), (2.5, 4), ("2", 4)])
def test_resize_rejects_invalid_dimensions(dimensions):
    with pytest.raises(HeddleError, match="positive integer"):
        lookup("resize").func(make_clip(), *dimensions)


# ---------------------------------------------------------------------------
# transforms: reverse
# ---------------------------------------------------------------------------


def test_reverse_matches_negative_one_speed():
    clip = Clip(
        [solid((1, 0, 0, 255)), solid((2, 0, 0, 255))],
        [60, 140],
        4,
    )

    out = lookup("reverse").func(clip)
    speed_out = apply_speed(clip, -1)

    assert out.durations == speed_out.durations == [140, 60]
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (2, 0, 0, 255),
        (1, 0, 0, 255),
    ]
    assert out.loop == 4


def test_reverse_call_works_as_pipeline_stage():
    clip = Clip(
        [solid((1, 0, 0, 255)), solid((2, 0, 0, 255))],
        [100, 200],
        0,
    )

    out = eval_node(expr_of("reverse()"), clip, Env())

    assert out.durations == [200, 100]


# ---------------------------------------------------------------------------
# transforms: text
# ---------------------------------------------------------------------------


def _non_black_bbox(frame):
    mask = Image.new("1", frame.size)
    mask.putdata(
        [pixel[:3] != (0, 0, 0) for pixel in frame.get_flattened_data()]
    )
    return mask.getbbox()


def test_text_draws_at_bare_named_position_and_preserves_metadata():
    clip = make_clip(n=2, color=(0, 0, 0, 255), dur=65, size=(30, 30))
    clip.loop = 3

    out = eval_node(expr_of('text("X", pos=BR)'), clip, Env())

    left, top, right, bottom = _non_black_bbox(out.frames[0])
    assert left >= 20
    assert top >= 20
    assert right <= 30
    assert bottom <= 30
    assert out.durations == [65, 65]
    assert out.loop == 3


def test_text_supports_descriptive_top_left_position():
    out = lookup("text").func(
        make_clip(color=(0, 0, 0, 255), size=(30, 30)), "X", "top-left"
    )

    left, top, right, bottom = _non_black_bbox(out.frames[0])
    assert left == 0
    assert top == 0
    assert right < 15
    assert bottom < 15


def test_text_rejects_unknown_position():
    with pytest.raises(HeddleError, match="unknown text position"):
        lookup("text").func(make_clip(), "hello", "somewhere")


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
# transforms: apply_concat
# ---------------------------------------------------------------------------


def test_apply_concat_appends_frames_and_durations():
    first = Clip([solid((1, 0, 0, 255)), solid((2, 0, 0, 255))], [100, 200], 3)
    second = Clip([solid((3, 0, 0, 255))], [300], 0)

    out = apply_concat([first, second])

    assert out.durations == [100, 200, 300]
    assert out.loop == 3
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (1, 0, 0, 255),
        (2, 0, 0, 255),
        (3, 0, 0, 255),
    ]


def test_dissolve_inserts_blended_frames_with_exact_duration():
    first = make_clip(color=(255, 0, 0, 255), dur=100)
    second = make_clip(color=(0, 0, 255, 255), dur=200)

    out = apply_concat([first, Dissolve(250), second])

    assert out.durations == [100, 100, 100, 50, 200]
    assert sum(out.durations) == 550
    for frame in out.frames[1:4]:
        red, green, blue, alpha = frame.getpixel((0, 0))
        assert red > 0
        assert green == 0
        assert blue > 0
        assert alpha == 255


def test_dissolve_requires_matching_dimensions():
    with pytest.raises(HeddleError, match="same dimensions"):
        apply_concat(
            [
                make_clip(size=(2, 2)),
                Dissolve(100),
                make_clip(size=(3, 2)),
            ]
        )


@pytest.mark.parametrize(
    "items",
    [
        [Dissolve(100), make_clip()],
        [make_clip(), Dissolve(100)],
        [make_clip(), Dissolve(100), Dissolve(100), make_clip()],
    ],
)
def test_dissolve_must_appear_between_clips(items):
    with pytest.raises(HeddleError, match="between two clips"):
        apply_concat(items)


def test_dissolve_function_builds_transition_in_seconds():
    transition = lookup("dissolve").func(0.25)

    assert transition == Dissolve(250)
    assert not lookup("dissolve").needs_input


@pytest.mark.parametrize("sec", [0, -1, float("inf"), "slow"])
def test_dissolve_rejects_invalid_duration(sec):
    with pytest.raises(HeddleError, match="positive finite"):
        lookup("dissolve").func(sec)


# ---------------------------------------------------------------------------
# transforms: apply_overlay
# ---------------------------------------------------------------------------


def test_apply_overlay_composites_left_layer_over_right():
    top = make_clip(color=(255, 0, 0, 255))
    bottom = make_clip(color=(0, 0, 255, 255))

    out = apply_overlay([top, bottom])

    assert out.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)


def test_apply_overlay_repeats_static_layer_over_animation():
    top = make_clip(color=(255, 0, 0, 255))
    bottom = Clip(
        [solid((0, 0, 255, 255)), solid((0, 255, 0, 255))],
        [100, 200],
        0,
    )

    out = apply_overlay([top, bottom])

    assert out.durations == [100, 200]
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (255, 0, 0, 255),
        (255, 0, 0, 255),
    ]


def test_apply_overlay_rejects_mismatched_animated_timing():
    first = make_clip(n=2, dur=100)
    second = make_clip(n=2, dur=200)

    with pytest.raises(HeddleError):
        apply_overlay([first, second])


# ---------------------------------------------------------------------------
# transforms: apply_stack
# ---------------------------------------------------------------------------


def test_apply_stack_horizontal_places_right_item_after_left():
    left = make_clip(color=(255, 0, 0, 255), size=(1, 1))
    right = make_clip(color=(0, 0, 255, 255), size=(2, 1))

    out = apply_stack([left, right], [("h", None)])

    assert out.frames[0].size == (3, 1)
    assert out.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert out.frames[0].getpixel((1, 0)) == (0, 0, 255, 255)
    assert out.frames[0].getpixel((2, 0)) == (0, 0, 255, 255)


def test_apply_stack_vertical_places_lower_item_below_upper():
    upper = make_clip(color=(255, 0, 0, 255), size=(1, 1))
    lower = make_clip(color=(0, 0, 255, 255), size=(1, 2))

    out = apply_stack([upper, lower], [("v", None)])

    assert out.frames[0].size == (1, 3)
    assert out.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert out.frames[0].getpixel((0, 1)) == (0, 0, 255, 255)
    assert out.frames[0].getpixel((0, 2)) == (0, 0, 255, 255)


def test_apply_stack_rejects_modes_until_defined():
    with pytest.raises(HeddleError):
        apply_stack([make_clip(), make_clip()], [("h", "fit")])


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


def test_eval_concat_appends_source_clips(monkeypatch):
    clips = {
        "a.gif": Clip([solid((1, 0, 0, 255))], [100], 0),
        "b.gif": Clip([solid((2, 0, 0, 255))], [200], 0),
    }
    monkeypatch.setattr(
        interpreter, "resolve_source", lambda ident, cwd: f"{ident}.gif"
    )
    monkeypatch.setattr(interpreter, "load", lambda path: clips[path])

    out = eval_node(expr_of("a >> b"), None, Env())

    assert out.durations == [100, 200]
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (1, 0, 0, 255),
        (2, 0, 0, 255),
    ]


def test_eval_concat_renders_dissolve_function(monkeypatch):
    clips = {
        "a.gif": make_clip(color=(255, 0, 0, 255), dur=100),
        "b.gif": make_clip(color=(0, 0, 255, 255), dur=200),
    }
    monkeypatch.setattr(
        interpreter, "resolve_source", lambda ident, cwd: f"{ident}.gif"
    )
    monkeypatch.setattr(interpreter, "load", lambda path: clips[path])

    out = eval_node(expr_of("a >> dissolve(0.2) >> b"), None, Env())

    assert out.durations == [100, 100, 100, 200]
    assert out.frames[1].getpixel((0, 0)) == (191, 0, 63, 255)
    assert out.frames[2].getpixel((0, 0)) == (63, 0, 191, 255)


def test_eval_concat_as_grouped_pipeline_stage(monkeypatch):
    img = Image.new("RGBA", (2, 1))
    img.putpixel((0, 0), (255, 0, 0, 255))
    img.putpixel((1, 0), (0, 255, 0, 255))
    src_clip = Clip([img], [100], 0)
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "im.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: src_clip)

    out = eval_node(expr_of("im | (hflip >> hflip^2)"), None, Env())

    assert out.durations == [100, 50]
    assert out.frames[0].getpixel((0, 0)) == (0, 255, 0, 255)
    assert out.frames[1].getpixel((0, 0)) == (0, 255, 0, 255)


def test_eval_overlay_composites_sources(monkeypatch):
    clips = {
        "top.gif": make_clip(color=(255, 0, 0, 255)),
        "bottom.gif": make_clip(color=(0, 0, 255, 255)),
    }
    monkeypatch.setattr(
        interpreter, "resolve_source", lambda ident, cwd: f"{ident}.gif"
    )
    monkeypatch.setattr(interpreter, "load", lambda path: clips[path])

    out = eval_node(expr_of("top over bottom"), None, Env())

    assert out.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)


def test_eval_stack_mixed_layout(monkeypatch):
    clips = {
        "a.gif": make_clip(color=(255, 0, 0, 255), size=(1, 1)),
        "b.gif": make_clip(color=(0, 0, 255, 255), size=(1, 1)),
        "c.gif": make_clip(color=(0, 255, 0, 255), size=(2, 1)),
    }
    monkeypatch.setattr(
        interpreter, "resolve_source", lambda ident, cwd: f"{ident}.gif"
    )
    monkeypatch.setattr(interpreter, "load", lambda path: clips[path])

    out = eval_node(expr_of("a & b / c"), None, Env())

    assert out.frames[0].size == (2, 2)
    assert out.frames[0].getpixel((0, 0)) == (255, 0, 0, 255)
    assert out.frames[0].getpixel((1, 0)) == (0, 0, 255, 255)
    assert out.frames[0].getpixel((0, 1)) == (0, 255, 0, 255)
    assert out.frames[0].getpixel((1, 1)) == (0, 255, 0, 255)


def test_eval_stack_mode_raises(monkeypatch):
    monkeypatch.setattr(interpreter, "resolve_source", lambda ident, cwd: "x.gif")
    monkeypatch.setattr(interpreter, "load", lambda path: make_clip())

    with pytest.raises(HeddleError):
        eval_node(expr_of("a &fit b"), None, Env())


# ---------------------------------------------------------------------------
# indexing / slicing
# ---------------------------------------------------------------------------


def test_index_single_frame_uses_default_static_duration():
    clip = Clip(
        [solid((1, 0, 0, 255)), solid((2, 0, 0, 255))],
        [250, 500],
        0,
    )

    out = eval_node(expr_of("src[1]"), None, Env(names={"src": clip}))

    assert out.durations == [100]
    assert out.frames[0].getpixel((0, 0)) == (2, 0, 0, 255)


def test_index_frame_slice_preserves_original_durations():
    clip = Clip(
        [
            solid((1, 0, 0, 255)),
            solid((2, 0, 0, 255)),
            solid((3, 0, 0, 255)),
        ],
        [100, 200, 300],
        0,
    )

    out = eval_node(expr_of("src[1:]"), None, Env(names={"src": clip}))

    assert out.durations == [200, 300]
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (2, 0, 0, 255),
        (3, 0, 0, 255),
    ]


def test_index_time_offsets_trim_edge_durations():
    clip = Clip(
        [
            solid((1, 0, 0, 255)),
            solid((2, 0, 0, 255)),
            solid((3, 0, 0, 255)),
        ],
        [100, 200, 300],
        0,
    )

    out = eval_node(expr_of("src[50ms:350ms]"), None, Env(names={"src": clip}))

    assert out.durations == [50, 200, 50]
    assert [frame.getpixel((0, 0)) for frame in out.frames] == [
        (1, 0, 0, 255),
        (2, 0, 0, 255),
        (3, 0, 0, 255),
    ]


def test_index_time_offset_single_frame_uses_default_static_duration():
    clip = Clip(
        [solid((1, 0, 0, 255)), solid((2, 0, 0, 255))],
        [100, 200],
        0,
    )

    out = eval_node(expr_of("src[150ms]"), None, Env(names={"src": clip}))

    assert out.durations == [100]
    assert out.frames[0].getpixel((0, 0)) == (2, 0, 0, 255)


def test_index_spatial_axes_are_y_then_x():
    img = Image.new("RGBA", (3, 2))
    for y in range(2):
        for x in range(3):
            img.putpixel((x, y), (x, y, 0, 255))
    clip = Clip([img], [100], 0)

    out = eval_node(expr_of("src[:, 1, 1:3]"), None, Env(names={"src": clip}))

    assert out.frames[0].size == (2, 1)
    assert [out.frames[0].getpixel((x, 0)) for x in range(2)] == [
        (1, 1, 0, 255),
        (2, 1, 0, 255),
    ]


def test_index_slice_clamps_but_single_index_raises():
    clip = make_clip(n=2)

    out = eval_node(expr_of("src[0:999]"), None, Env(names={"src": clip}))
    assert len(out.frames) == 2

    with pytest.raises(HeddleError):
        eval_node(expr_of("src[999]"), None, Env(names={"src": clip}))


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


def test_gif_output_defaults_to_looping_forever(tmp_path):
    clip = Clip(
        [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))],
        [120, 180],
        1,
    )
    path = tmp_path / "out.gif"
    save(clip, str(path))
    assert path.exists()

    reloaded = load(str(path))
    assert len(reloaded.frames) == 2
    assert reloaded.durations == [120, 180]
    assert reloaded.loop == 0


def test_gif_output_accepts_explicit_loop_count(tmp_path):
    clip = Clip(
        [solid((255, 0, 0, 255)), solid((0, 0, 255, 255))],
        [100, 100],
    )
    path = tmp_path / "out.gif"

    save(clip, str(path), loop=2)

    assert load(str(path)).loop == 2
