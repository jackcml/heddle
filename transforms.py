from __future__ import annotations

from dataclasses import dataclass
import math

from PIL import Image, ImageFilter, ImageOps

from clip import DEFAULT_MS, Clip
from errors import HeddleError
from registry import transform

# This module is imported for its side effects: each @transform call registers a
# built-in op. New named transforms go here as decorated functions.


@transform("hflip")
def hflip(clip: Clip) -> Clip:
    """Mirror every frame left-to-right."""
    return Clip(
        [ImageOps.mirror(frame) for frame in clip.frames],
        list(clip.durations),
        clip.loop,
    )


@transform("blur", params=("stdev",))
def blur(clip: Clip, stdev: float) -> Clip:
    """Apply a Gaussian blur with standard deviation `stdev` to every frame."""
    if not isinstance(stdev, (int, float)) or stdev < 0:
        raise HeddleError("blur stdev must be a non-negative number")
    return Clip(
        [frame.filter(ImageFilter.GaussianBlur(stdev)) for frame in clip.frames],
        list(clip.durations),
        clip.loop,
    )


@transform("scale", params=("factor",))
def scale(clip: Clip, factor: float) -> Clip:
    """Resize every frame uniformly by `factor`."""
    if (
        not isinstance(factor, (int, float))
        or isinstance(factor, bool)
        or not math.isfinite(factor)
        or factor <= 0
    ):
        raise HeddleError("scale factor must be a positive finite number")

    frames = []
    for frame in clip.frames:
        size = (
            max(1, round(frame.width * factor)),
            max(1, round(frame.height * factor)),
        )
        frames.append(frame.resize(size, Image.Resampling.LANCZOS))
    return Clip(frames, list(clip.durations), clip.loop)


## Operators (symbolic, not named)


def apply_speed(clip: Clip, k: float) -> Clip:
    """Retime a clip by factor `k` via duration scaling (lossless).

    `^k` divides each per-frame duration by `k`, so the clip plays `k`x faster
    without dropping any frames. `k < 0` reverses playback; `k == 0` is undefined.
    """
    if k == 0:
        raise HeddleError("speed factor cannot be zero")

    frames = list(clip.frames)
    durations = list(clip.durations)
    if k < 0:
        frames.reverse()
        durations.reverse()
        k = -k

    scaled = [max(1, round(d / k)) for d in durations]  # floor at 1ms per frame
    return Clip(frames, scaled, clip.loop)


def apply_concat(clips: list[Clip]) -> Clip:
    """Append clips in time, preserving frame durations."""
    frames = []
    durations = []
    for clip in clips:
        frames.extend(clip.frames)
        durations.extend(clip.durations)
    return Clip(frames, durations, clips[0].loop if clips else 0)


def apply_overlay(layers: list[Clip]) -> Clip:
    """Composite layers in visual order: `a over b` places `a` on top of `b`."""
    frame_count, durations, loop = _shared_timeline(layers, "over")
    frames = []

    for i in range(frame_count):
        out = _frame_at(layers[-1], i).copy()
        for layer in reversed(layers[:-1]):
            top = _frame_at(layer, i)
            if top.size != out.size:
                raise HeddleError("over requires layers to have the same dimensions")
            out = Image.alpha_composite(out, top)
        frames.append(out)

    return Clip(frames, durations, loop)


def apply_stack(items: list[Clip], joins: list[tuple[str, object]]) -> Clip:
    """Lay clips out spatially, left-to-right for `&` and top-to-bottom for `/`."""
    for axis, mode in joins:
        if mode is not None:
            raise HeddleError(f"stack mode {mode!r} is not implemented yet")
        if axis not in ("h", "v"):
            raise HeddleError(f"unknown stack axis {axis!r}")

    frame_count, durations, loop = _shared_timeline(items, "stack")
    frames = []

    for i in range(frame_count):
        out = _frame_at(items[0], i).copy()
        for item, (axis, _) in zip(items[1:], joins):
            out = _stack_pair(out, _frame_at(item, i), axis)
        frames.append(out)

    return Clip(frames, durations, loop)


@dataclass(frozen=True)
class TimeIndex:
    offset_ms: int


@dataclass(frozen=True)
class TimeSlice:
    start_ms: int | None
    stop_ms: int | None
    step: int | None = None


def apply_index(clip: Clip, axes: list[object]) -> Clip:
    """Slice a clip as [time, y, x], copying selected frames."""
    time_axis, y_axis, x_axis = _complete_axes(axes)
    frames, durations = _select_time(clip, time_axis)
    frames = [_slice_frame(frame, y_axis, x_axis) for frame in frames]
    return Clip(frames, durations, clip.loop)


def _stack_pair(left: Image.Image, right: Image.Image, axis: str) -> Image.Image:
    if axis == "h":
        out = Image.new(
            "RGBA", (left.width + right.width, max(left.height, right.height))
        )
        out.alpha_composite(left, (0, 0))
        out.alpha_composite(right, (left.width, 0))
        return out

    out = Image.new("RGBA", (max(left.width, right.width), left.height + right.height))
    out.alpha_composite(left, (0, 0))
    out.alpha_composite(right, (0, left.height))
    return out


def _complete_axes(axes: list[object]) -> tuple[object, object, object]:
    if len(axes) > 3:
        raise HeddleError("indexing supports at most three axes: [time, y, x]")
    full = slice(None)
    return tuple((axes + [full, full, full])[:3])


def _select_time(clip: Clip, axis: object) -> tuple[list[Image.Image], list[int]]:
    if isinstance(axis, TimeIndex):
        idx = _frame_at_offset(clip, axis.offset_ms)
        return [clip.frames[idx].copy()], [DEFAULT_MS]

    if isinstance(axis, TimeSlice):
        return _slice_time_offsets(clip, axis)

    if isinstance(axis, int):
        idx = _normalize_index(axis, len(clip.frames), "frame")
        return [clip.frames[idx].copy()], [DEFAULT_MS]

    if isinstance(axis, slice):
        indices = range(*axis.indices(len(clip.frames)))
        return (
            [clip.frames[i].copy() for i in indices],
            [clip.durations[i] for i in indices],
        )

    raise HeddleError("invalid time axis selector")


def _slice_time_offsets(
    clip: Clip, axis: TimeSlice
) -> tuple[list[Image.Image], list[int]]:
    if axis.step is not None and axis.step <= 0:
        raise HeddleError("time-offset slices require a positive frame step")

    total = sum(clip.durations)
    start = _normalize_offset(axis.start_ms, total, 0)
    stop = _normalize_offset(axis.stop_ms, total, total)

    selected_frames = []
    selected_durations = []
    elapsed = 0
    for frame, duration in zip(clip.frames, clip.durations):
        frame_start = elapsed
        frame_stop = elapsed + duration
        elapsed = frame_stop

        overlap_start = max(frame_start, start)
        overlap_stop = min(frame_stop, stop)
        if overlap_start < overlap_stop:
            selected_frames.append(frame.copy())
            selected_durations.append(overlap_stop - overlap_start)

    if axis.step is not None:
        selected_frames = selected_frames[:: axis.step]
        selected_durations = selected_durations[:: axis.step]
    return selected_frames, selected_durations


def _slice_frame(frame: Image.Image, y_axis: object, x_axis: object) -> Image.Image:
    x_indices = _spatial_indices(x_axis, frame.width, "x")
    y_indices = _spatial_indices(y_axis, frame.height, "y")

    if not x_indices or not y_indices:
        raise HeddleError("spatial slice selected no pixels")

    out = Image.new("RGBA", (len(x_indices), len(y_indices)))
    out_px = out.load()
    in_px = frame.load()
    for out_y, in_y in enumerate(y_indices):
        for out_x, in_x in enumerate(x_indices):
            out_px[out_x, out_y] = in_px[in_x, in_y]
    return out


def _spatial_indices(axis: object, size: int, name: str) -> list[int]:
    if isinstance(axis, int):
        return [_normalize_index(axis, size, name)]
    if isinstance(axis, slice):
        return list(range(*axis.indices(size)))
    raise HeddleError(f"invalid {name} axis selector")


def _frame_at_offset(clip: Clip, offset_ms: int) -> int:
    total = sum(clip.durations)
    offset = offset_ms + total if offset_ms < 0 else offset_ms
    if offset < 0 or offset >= total:
        raise HeddleError("time index out of range")

    elapsed = 0
    for idx, duration in enumerate(clip.durations):
        elapsed += duration
        if offset < elapsed:
            return idx
    raise HeddleError("time index out of range")


def _normalize_offset(offset_ms: int | None, total: int, default: int) -> int:
    if offset_ms is None:
        return default
    offset = offset_ms + total if offset_ms < 0 else offset_ms
    return min(max(offset, 0), total)


def _normalize_index(index: int, size: int, name: str) -> int:
    normalized = index + size if index < 0 else index
    if normalized < 0 or normalized >= size:
        raise HeddleError(f"{name} index out of range")
    return normalized


def _shared_timeline(clips: list[Clip], op: str) -> tuple[int, list[int], int]:
    animated = [clip for clip in clips if clip.is_animated]
    if not animated:
        return 1, [max(clip.durations[0] for clip in clips)], clips[0].loop

    ref = animated[0]
    for clip in animated[1:]:
        if len(clip.frames) != len(ref.frames) or clip.durations != ref.durations:
            raise HeddleError(
                f"{op} requires animated inputs to share the same frame timing"
            )
    return len(ref.frames), list(ref.durations), ref.loop


def _frame_at(clip: Clip, index: int) -> Image.Image:
    return clip.frames[0] if not clip.is_animated else clip.frames[index]
