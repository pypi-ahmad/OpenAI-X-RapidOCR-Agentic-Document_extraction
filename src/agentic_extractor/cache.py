"""Bounded process-memory caches for local derived document data.

Responsible for the LRU eviction mechanism and the pixel-content hash used to
key entries. It does NOT decide what belongs in a cache, or enforce that a
cache key captures every input that can change a result (exact rendered
pixels plus engine/model/version signature) — that invariant is owned by each
caller (`ingest.py` for renders, `ocr.py`/`layout.py`/`table_structure.py` for
model outputs), which must never key on something looser than the content
actually consumed. Caches are process-memory only: they are never written to
disk and disappear on process exit. Next: `ingest.py` for the first concrete
use of `page_image_hash`."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock

from PIL import Image


def page_image_hash(image: Image.Image) -> str:
    """Return a stable hash of the exact RGB pixels consumed by local models."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"{rgb.width}x{rgb.height}:RGB\0".encode())
    digest.update(rgb.tobytes())
    return digest.hexdigest()


class ByteLRUCache:
    """Store immutable byte payloads within entry and byte limits.

    All methods take `self._lock` (an `RLock`), so a single instance is safe
    to share across the OCR worker threads in `ocr.py` without external
    synchronization. Values must be treated as immutable once stored: callers
    hand back the same bytes object on a hit, so mutating it would corrupt
    every other reader of that cache entry.
    """

    def __init__(self, *, max_bytes: int, max_entries: int) -> None:
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._values: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = RLock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def put(self, key: str, value: bytes) -> bool:
        if len(value) > self.max_bytes:
            return False
        with self._lock:
            previous = self._values.pop(key, None)
            if previous is not None:
                self._bytes -= len(previous)
            self._values[key] = value
            self._bytes += len(value)
            while self._bytes > self.max_bytes or len(self._values) > self.max_entries:
                _, evicted = self._values.popitem(last=False)
                self._bytes -= len(evicted)
        return True

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._bytes = 0


_CACHE_BYTES = 128 * 1024 * 1024
RENDER_CACHE = ByteLRUCache(max_bytes=_CACHE_BYTES, max_entries=256)
OCR_CACHE = ByteLRUCache(max_bytes=_CACHE_BYTES, max_entries=256)
LAYOUT_CACHE = ByteLRUCache(max_bytes=_CACHE_BYTES, max_entries=256)
TABLE_CACHE = ByteLRUCache(max_bytes=_CACHE_BYTES, max_entries=256)


def clear_local_caches() -> None:
    """Clear process caches explicitly for tests and controlled maintenance."""
    RENDER_CACHE.clear()
    OCR_CACHE.clear()
    LAYOUT_CACHE.clear()
    TABLE_CACHE.clear()
