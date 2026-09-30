"""Typed references (ADR-0010): an Object reference is ``obj_…``, a Run
reference is ``run_…``. The kind is in the value, so a reference passed to the
wrong kind of tool is refused as such instead of reported "not found".

Cache ids stay internal and untyped: they never cross the tool boundary.
"""

from __future__ import annotations

import re
import uuid

OBJECT = "obj_"
RUN = "run_"

_NAMES = {OBJECT: "Object reference", RUN: "Run reference"}
_HINTS = {
    RUN: "get_run_summary(run_ref) returns the object_ref that run produced",
    OBJECT: "use inspect_object(object_ref) for objects",
}


def new(kind: str) -> str:
    """A fresh reference of ``kind`` (OBJECT or RUN)."""
    return f"{kind}{uuid.uuid4().hex[:8]}"


def wrong_kind(ref: object, expected: str) -> str | None:
    """An error message when ``ref`` is a reference of the other kind, else None."""
    if not isinstance(ref, str):
        return None
    for kind, name in _NAMES.items():
        if kind != expected and re.fullmatch(f"{kind}[0-9a-f]{{8}}", ref):
            return (f"'{ref}' is a {name}, but this takes a {_NAMES[expected]} "
                    f"({expected}…): {_HINTS[kind]}.")
    return None
