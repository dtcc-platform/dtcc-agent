"""In-memory object store for dtcc-core objects.

Stores intermediate results (PointCloud, Mesh, Raster, etc.) with short
Object references (``obj_…``, ADR-0010) so that multi-step pipelines can
reference previous outputs.
Thread-safe via a lock; LRU eviction keeps memory bounded, per store and
across every store sharing a MemoryBudget (T11, #23).
"""

from __future__ import annotations

import itertools
import sys
import threading
import time
import types
import weakref
from typing import Any

import numpy as np

from . import refs

# Never walked into: shared by everything, owned by nothing stored.
_OPAQUE = (type, types.ModuleType, types.FunctionType, types.BuiltinFunctionType,
           types.MethodType)
_LEAVES = (str, bytes, bytearray, int, float, complex, bool, type(None))


def _estimate_bytes(obj: Any) -> int:
    """Bytes `obj` holds, counting everything it reaches once (U4, #13).

    Walks dicts, sequences and object attributes, so GeoJSON-like dicts, lists
    of Buildings and Core geometry count their contents. A numpy array counts
    its buffer, a view its base, and a simulation result its ``.x.array``.
    """
    total = 0
    seen: set[int] = set()
    stack = [obj]
    while stack:
        item = stack.pop()
        if id(item) in seen or isinstance(item, _OPAQUE):
            continue
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            if isinstance(item.base, np.ndarray):
                stack.append(item.base)
            else:
                total += item.nbytes
            continue
        total += sys.getsizeof(item)
        if isinstance(item, _LEAVES):
            continue
        if isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            stack.extend(item)
        else:
            attrs = getattr(item, "__dict__", None)
            if isinstance(attrs, dict):
                stack.append(attrs)
            for cls in type(item).__mro__:
                for slot in getattr(cls, "__slots__", ()):
                    if isinstance(slot, str) and hasattr(item, slot):
                        stack.append(getattr(item, slot))
            # dolfinx keeps a Function's values behind a property, not an attribute.
            values = getattr(getattr(item, "x", None), "array", None)
            if isinstance(values, np.ndarray):
                stack.append(values)
    return max(total, 64)


class MemoryBudget:
    """What every Session's ObjectStore may hold together (T11, #23).

    Stores sharing a budget share its lock. When their total passes
    `max_bytes`, the least recently used Object in any of them is evicted, so
    memory an idle Session holds goes to the Sessions in use.
    """

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.total_bytes = 0
        self.lock = threading.Lock()
        self._stores: weakref.WeakSet[ObjectStore] = weakref.WeakSet()
        self._clock = itertools.count()

    def tick(self) -> int:
        """The next access time; a counter, so ties cannot happen."""
        return next(self._clock)

    def evict_if_needed(self) -> None:
        """Evict the least recently used Object anywhere until under budget.
        Must be called with the lock held."""
        while self.total_bytes > self.max_bytes:
            candidates = [
                (entry["last_accessed"], store, obj_id)
                for store in self._stores
                for obj_id, entry in store._objects.items()
            ]
            if not candidates:
                return
            _, store, obj_id = min(candidates, key=lambda c: c[0])
            store._remove(obj_id)


class ObjectStore:
    """Thread-safe in-memory store for dtcc-core objects.

    Parameters
    ----------
    max_bytes : int
        The most this store may hold. Least recently used Objects are evicted
        past it, and a single Object larger than it is not kept. Default 2 GB.
    budget : MemoryBudget, optional
        A budget shared with other stores. Without one the store has its own.
    """

    def __init__(self, max_bytes: int = 2 * 1024**3, budget: MemoryBudget | None = None):
        self._budget = budget or MemoryBudget(max_bytes)
        self._lock = self._budget.lock
        self._objects: dict[str, dict[str, Any]] = {}
        self._max_bytes = max_bytes
        self._total_bytes = 0
        with self._lock:
            self._budget._stores.add(self)

    def store(self, obj: Any, source_op: str = "", label: str = "") -> str | None:
        """Store an object and return its Object reference.

        Returns None, keeping nothing, when the object alone is larger than
        this store may hold (U4): evicting everything else would not make room.
        """
        nbytes = _estimate_bytes(obj)
        if nbytes > self._max_bytes:
            return None
        obj_id = refs.new(refs.OBJECT)
        with self._lock:
            self._objects[obj_id] = {
                "object": obj,
                "type": type(obj).__name__,
                "source_op": source_op,
                "label": label,
                "created": time.time(),
                "last_accessed": self._budget.tick(),
                "nbytes": nbytes,
            }
            self._total_bytes += nbytes
            self._budget.total_bytes += nbytes
            self._evict_if_needed()
            self._budget.evict_if_needed()
        return obj_id

    def not_stored(self) -> str:
        """Why an Object this store refused has no object_ref."""
        return (
            f"Too large to keep: larger than the {self._max_bytes // 1024**2} MB this "
            "Session may hold, so it has no object_ref for later steps. "
            "Try a smaller area."
        )

    def get(self, obj_id: str) -> Any:
        """Retrieve an object by ID. Raises KeyError if not found."""
        with self._lock:
            if obj_id not in self._objects:
                raise KeyError(f"Object '{obj_id}' not found in store")
            self._objects[obj_id]["last_accessed"] = self._budget.tick()
            return self._objects[obj_id]["object"]

    def delete(self, obj_id: str) -> dict[str, Any] | None:
        """Remove an object by ID and return its entry, or None if absent."""
        with self._lock:
            return self._remove(obj_id)

    def clear(self) -> None:
        """Drop every Object, returning their bytes to the shared budget."""
        with self._lock:
            for obj_id in list(self._objects):
                self._remove(obj_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return summaries of stored objects, most recent first."""
        with self._lock:
            entries = sorted(
                self._objects.items(),
                key=lambda kv: kv[1]["created"],
                reverse=True,
            )
            result = []
            for obj_id, entry in entries[:limit]:
                result.append({
                    "object_ref": obj_id,
                    "type": entry["type"],
                    "source_op": entry["source_op"],
                    "label": entry["label"],
                    "created": entry["created"],
                    "nbytes": entry["nbytes"],
                })
        return result

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    def __len__(self) -> int:
        return len(self._objects)

    def __contains__(self, obj_id: str) -> bool:
        return obj_id in self._objects

    def _remove(self, obj_id: str) -> dict[str, Any] | None:
        """Must be called with the lock held."""
        entry = self._objects.pop(obj_id, None)
        if entry is not None:
            self._total_bytes -= entry["nbytes"]
            self._budget.total_bytes -= entry["nbytes"]
        return entry

    def _evict_if_needed(self) -> None:
        """Evict least recently used Objects until under this store's cap.
        Must be called with the lock held."""
        while self._total_bytes > self._max_bytes and self._objects:
            lru_id = min(self._objects, key=lambda k: self._objects[k]["last_accessed"])
            self._remove(lru_id)
