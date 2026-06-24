from __future__ import annotations

from PIL import ImageOps

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
