from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

# Marks required arguments with no default, so `None` can be a valid default value
_REQUIRED = object()


@dataclass
class Param:
    """A single transform parameter: its name and optional default value."""

    name: str
    default: object = _REQUIRED

    @property
    def required(self) -> bool:
        return self.default is _REQUIRED


@dataclass
class Transform:
    """A named operation invoked as `func(clip, **bound_params) -> Clip`.

    `needs_input` is False for programmatic sources that produce a clip
    from nothing rather than transforming an incoming one.
    """

    name: str
    func: Callable
    params: tuple = ()  # tuple[Param, ...], in positional order
    needs_input: bool = True


_REGISTRY: dict[str, Transform] = {}


def transform(name: str, params=(), needs_input: bool = True):
    """Decorator registering `func` under `name`.

    Adding an operator or function later is one decorated function here,
    with no interpreter changes. `params` entries may be plain names (required)
    or `Param` instances (to supply defaults).
    """
    norm = tuple(p if isinstance(p, Param) else Param(p) for p in params)

    def register(func: Callable) -> Callable:
        _REGISTRY[name] = Transform(name, func, norm, needs_input)
        return func

    return register


def lookup(name: str) -> Optional[Transform]:
    return _REGISTRY.get(name)
