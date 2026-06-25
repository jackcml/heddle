from __future__ import annotations

from PIL import Image, ImageOps

from clip import Clip
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
