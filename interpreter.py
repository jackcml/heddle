from __future__ import annotations

from dataclasses import dataclass, field

import transforms  # noqa: F401 -- imported for transform-registration side effects
from clip import Clip, load, resolve_source, save
from errors import HeddleError
from parser import (
    Binding,
    Call,
    Concat,
    Index,
    Name,
    Num,
    Overlay,
    Pipe,
    Pipeline,
    Program,
    Slice,
    Sink,
    Speed,
    Stack,
    Str,
)
from registry import lookup
from transforms import (
    TimeIndex,
    TimeSlice,
    apply_concat,
    apply_index,
    apply_overlay,
    apply_speed,
    apply_stack,
)


@dataclass
class Env:
    """Execution state: the working directory plus named bindings."""

    cwd: str = "."
    names: dict[str, Clip] = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Statement execution
# ----------------------------------------------------------------------------


def run_program(program: Program, cwd: str = ".") -> Env:
    """Run statements in order; return the resulting environment."""
    env = Env(cwd=cwd)
    for stmt in program.statements:
        if isinstance(stmt, Binding):
            env.names[stmt.name] = run_binding(stmt, env)
        else:
            run_pipeline(stmt, env)
    return env


def run_binding(binding: Binding, env: Env) -> Clip:
    pl = binding.pipeline
    clip = eval_node(pl.expr, None, env)

    # A binding that carries a sink (`x = a | b > "f"`) both names the clip and writes it.
    if pl.sink is not None:
        save(clip, pl.sink.target.value)
    return clip


def run_pipeline(pl: Pipeline, env: Env) -> Clip:
    # A standalone pipeline must produce output or it does nothing observable.
    clip = eval_node(pl.expr, None, env)
    if pl.sink is None:
        raise HeddleError(
            'pipeline has no output; add a sink, e.g. > "out.gif"', pl.line, pl.col
        )
    save(clip, pl.sink.target.value)
    return clip


# ----------------------------------------------------------------------------
# Expression evaluation
# ----------------------------------------------------------------------------


def eval_node(node, input, env: Env) -> Clip:
    """Evaluate an expression to a Clip.

    `input` is the clip flowing in from the left (None at the head of a pipeline
    or binding). Dispatch by node type.
    """
    match node:
        case Name():
            return _eval_name(node, input, env)
        case Pipe(stages=stages):
            acc = input
            for stage in stages:
                acc = eval_node(stage, acc, env)
            return acc
        case Speed(base=base, factor=factor):
            return apply_speed(
                eval_node(base, input, env), eval_speed_factor(factor, env)
            )
        case Call():
            return eval_call(node, input, env)
        case Num() | Str():
            raise HeddleError("expected media, got a scalar value", node.line, node.col)

        case Concat(clips=clips):
            return apply_concat([eval_node(clip, input, env) for clip in clips])
        case Overlay(layers=layers):
            return apply_overlay([eval_node(layer, input, env) for layer in layers])
        case Stack(items=items, joins=joins):
            return apply_stack(
                [eval_node(item, input, env) for item in items],
                [(join.axis, join.mode) for join in joins],
            )
        case Index():
            return eval_index(node, input, env)
        case _:
            raise HeddleError(
                f"cannot evaluate {type(node).__name__}", node.line, node.col
            )


def _eval_name(node: Name, input, env: Env) -> Clip:
    name = node.ident
    if input is None:
        # If we're at the head, name = a source. A binding may shadow a file of the same name.
        if name in env.names:
            return env.names[name]
        return load(resolve_source(name, env.cwd))
    # Otherwise, name = a transform applied to the incoming clip.
    t = lookup(name)
    if t is None:
        raise HeddleError(f"unknown transform {name!r}", node.line, node.col)
    return _apply(t, input, [], node, env)


def eval_call(node: Call, input, env: Env) -> Clip:
    func = node.func
    if not isinstance(func, Name):
        raise HeddleError("only named transforms can be called", func.line, func.col)
    t = lookup(func.ident)
    if t is None:
        raise HeddleError(f"unknown function {func.ident!r}", func.line, func.col)
    return _apply(t, input, node.args, node, env)


def eval_index(node: Index, input, env: Env) -> Clip:
    clip = eval_node(node.base, input, env)
    axes = [_eval_axis(axis, idx, env, clip) for idx, axis in enumerate(node.axes)]
    return apply_index(clip, axes)


def _apply(t, input, args, node, env: Env) -> Clip:
    bound = _bind_args(t, args, node, env)
    if t.needs_input:
        if input is None:
            raise HeddleError(f"{t.name!r} needs an input clip", node.line, node.col)
        return t.func(input, **bound)
    return t.func(**bound)


def _bind_args(t, args, node, env: Env) -> dict:
    """Match call arguments to a transform's params, positional then keyword."""
    params = t.params
    bound: dict = {}
    seen_kw = False
    for idx, arg in enumerate(args):
        if arg.name is None:
            if seen_kw:
                raise HeddleError(
                    "positional argument after keyword argument", arg.line, arg.col
                )
            if idx >= len(params):
                raise HeddleError(
                    f"{t.name!r} takes at most {len(params)} argument(s)",
                    arg.line,
                    arg.col,
                )
            bound[params[idx].name] = eval_scalar(arg.value, env)
        else:
            seen_kw = True
            if not any(p.name == arg.name for p in params):
                raise HeddleError(
                    f"{t.name!r} has no parameter {arg.name!r}", arg.line, arg.col
                )
            if arg.name in bound:
                raise HeddleError(f"duplicate argument {arg.name!r}", arg.line, arg.col)
            bound[arg.name] = eval_scalar(arg.value, env)

    for p in params:
        if p.name not in bound:
            if p.required:
                raise HeddleError(
                    f"{t.name!r} is missing required argument {p.name!r}",
                    node.line,
                    node.col,
                )
            bound[p.name] = p.default
    return bound


# ----------------------------------------------------------------------------
# Index/slice axis evaluation
# ----------------------------------------------------------------------------


def _eval_axis(node, axis: int, env: Env, clip: Clip):
    if axis >= 3:
        raise HeddleError(
            "indexing supports at most three axes: [time, y, x]", node.line, node.col
        )
    if isinstance(node, Slice):
        return _eval_slice_axis(node, axis, env, clip)
    return _eval_single_axis(node, axis, env)


def _eval_single_axis(node, axis: int, env: Env):
    value, unit = _eval_axis_value(node, env)
    if axis == 0 and unit in ("ms", "s"):
        return TimeIndex(_duration_ms(value, unit, node))
    return _int_axis_value(
        value, unit, node, _axis_name(axis), allow_frame_unit=axis == 0
    )


def _eval_slice_axis(node: Slice, axis: int, env: Env, clip: Clip):
    start = _eval_optional_axis_value(node.start, env)
    stop = _eval_optional_axis_value(node.stop, env)
    step = _eval_optional_axis_value(node.step, env)

    if axis == 0:
        uses_time_offsets = _uses_time_unit(start) or _uses_time_unit(stop)
        if uses_time_offsets:
            return TimeSlice(
                _time_boundary_ms(start, clip, node),
                _time_boundary_ms(stop, clip, node),
                _optional_frame_step(step, node),
            )
        return slice(
            _optional_int(start, node, "frame", allow_frame_unit=True),
            _optional_int(stop, node, "frame", allow_frame_unit=True),
            _optional_int(step, node, "frame step", allow_frame_unit=True),
        )

    return slice(
        _optional_int(start, node, _axis_name(axis)),
        _optional_int(stop, node, _axis_name(axis)),
        _optional_int(step, node, f"{_axis_name(axis)} step"),
    )


def _eval_optional_axis_value(node, env: Env):
    if node is None:
        return None
    value, unit = _eval_axis_value(node, env)
    return value, unit, node


def _eval_axis_value(node, env: Env):
    if isinstance(node, Num):
        return node.value, node.unit
    return eval_scalar(node, env), None


def _uses_time_unit(value) -> bool:
    return value is not None and value[1] in ("ms", "s")


def _time_boundary_ms(value, clip: Clip, fallback_node):
    if value is None:
        return None

    raw, unit, node = value
    if unit in ("ms", "s"):
        return _duration_ms(raw, unit, node)
    frame = _int_axis_value(raw, unit, node, "frame", allow_frame_unit=True)
    if frame < 0:
        frame += len(clip.frames)
    frame = min(max(frame, 0), len(clip.frames))
    return sum(clip.durations[:frame])


def _optional_frame_step(value, fallback_node):
    if value is None:
        return None
    raw, unit, node = value
    if unit in ("ms", "s"):
        raise HeddleError(
            "time-offset slice step must be a frame count", node.line, node.col
        )
    return _int_axis_value(raw, unit, node, "frame step", allow_frame_unit=True)


def _optional_int(value, fallback_node, name: str, allow_frame_unit: bool = False):
    if value is None:
        return None
    raw, unit, node = value
    return _int_axis_value(raw, unit, node, name, allow_frame_unit)


def _int_axis_value(
    value, unit, node, name: str, allow_frame_unit: bool = False
) -> int:
    allowed_units = (None, "f") if allow_frame_unit else (None,)
    if unit not in allowed_units:
        raise HeddleError(f"{name} must be an integer index", node.line, node.col)
    if isinstance(value, float) and not value.is_integer():
        raise HeddleError(f"{name} must be an integer index", node.line, node.col)
    if not isinstance(value, (int, float)):
        raise HeddleError(f"{name} must be an integer index", node.line, node.col)
    return int(value)


def _duration_ms(value, unit, node) -> int:
    if unit == "s":
        ms = value * 1000
    elif unit == "ms":
        ms = value
    else:
        raise HeddleError("time offset must use 's' or 'ms'", node.line, node.col)
    return round(ms)


def _axis_name(axis: int) -> str:
    return ("time", "y", "x")[axis]


# ----------------------------------------------------------------------------
# Scalar evaluation (call arguments, speed factors)
# ----------------------------------------------------------------------------


def eval_scalar(node, env: Env):
    """Evaluate a non-media operand to an int/float/str.

    Unbound names are symbolic strings, which lets calls use readable enum-like
    arguments such as `text("hello", pos=TOP)`.
    """
    match node:
        case Num(value=value):
            return value
        case Str(value=value):
            return value
        case Name(ident=ident):
            if ident in env.names:
                val = env.names[ident]
                if isinstance(val, (int, float, str)):
                    return val
                raise HeddleError(
                    f"{ident!r} is not a scalar value", node.line, node.col
                )
            return ident
        case _:
            raise HeddleError(
                "expected a scalar value (number or string)", node.line, node.col
            )


def eval_speed_factor(node, env: Env) -> float:
    """Evaluate the operand of `^`, validating its unit.

    Unitless -> the factor itself; `%` -> fraction of normal speed (200% = 2x).
    `s` / `ms` / `f` are durations and frame counts, not a rate, so they error:
    true retiming, e.g., to exactly 2s would be separate functionality.
    """
    if isinstance(node, Num):
        if node.unit is None:
            return float(node.value)
        if node.unit == "%":
            return float(node.value) / 100.0
        raise HeddleError(
            f"speed factor must be a number or percent, not a {node.unit!r} value",
            node.line,
            node.col,
        )
    val = eval_scalar(node, env)
    if not isinstance(val, (int, float)):
        raise HeddleError("speed factor must be a number", node.line, node.col)
    return float(val)
