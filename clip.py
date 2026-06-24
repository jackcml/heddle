from __future__ import annotations

import os
from dataclasses import dataclass

from PIL import Image

from errors import HeddleError

# Default frame duration if a source declares none, should apply to static images
# NOTE: Reconsider / think about how to support holding images for n frames / seconds
DEFAULT_MS = 100

# in order of search priority
_SOURCE_EXTS = (".gif", ".png", ".jpg", ".jpeg", ".webp", ".bmp")


@dataclass
class Clip:
    """A media value: RGBA frames and per-frame durations (ms).

    A static image is a one-frame clip. Frames are stored as RGBA so future
    compositing (`over`, `&`, `/`) shares one pixel format. Final encoding
    happens at save time according to the output extension.
    """

    frames: list[Image.Image]
    durations: list[int]
    loop: int = 0  # animation loop count; 0 = loop forever

    @property
    def is_animated(self) -> bool:
        return len(self.frames) > 1


def load(path: str) -> Clip:
    """Read a source from disk into an RGBA Clip."""
    try:
        img = Image.open(path)
    except OSError as e:
        raise HeddleError(f"cannot open source {path!r}: {e}")

    loop = int(img.info.get("loop", 0))
    frames: list = []
    durations: list = []

    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        frames.append(img.convert("RGBA").copy())
        durations.append(int(img.info.get("duration", DEFAULT_MS)) or DEFAULT_MS)
    return Clip(frames, durations, loop)


def resolve_source(ident: str, cwd: str = ".") -> str:
    """Find file paths for bare source names (`im` -> `./im.gif`).

    Files in the working directory are exposed as globals named without their extension.
    The first existing extension in `_SOURCE_EXTS` takes priority.
    """
    for ext in _SOURCE_EXTS:
        candidate = os.path.join(cwd, ident + ext)
        if os.path.isfile(candidate):
            return candidate
    tried = ", ".join(ident + ext for ext in _SOURCE_EXTS)
    raise HeddleError(f"no source named {ident!r} in {cwd!r} (looked for {tried})")


def save(clip: Clip, path: str) -> None:
    """Write a Clip to disk with the output extension's encoding."""
    if not clip.frames:
        raise HeddleError("cannot save an empty clip")
    ext = os.path.splitext(path)[1].lower()

    if ext == ".gif":
        first, *rest = clip.frames
        # disposal=2 clears each frame before the next, avoiding ghosting
        first.save(
            path,
            save_all=True,
            append_images=rest,
            duration=clip.durations,
            loop=clip.loop,
            disposal=2,
        )
    else:
        # Still formats save only the first frame
        frame = clip.frames[0]
        if ext in (".jpg", ".jpeg"):
            frame = frame.convert("RGB")  # stripping alpha for JPEG
        frame.save(path)
