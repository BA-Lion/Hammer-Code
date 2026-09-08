from __future__ import annotations

from types import SimpleNamespace as NS


def event(type: str, **kwargs: object) -> NS:
    return NS(type=type, **kwargs)
