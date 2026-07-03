"""Response compression utilities for the exo master API.

Supports gzip (stdlib) and brotli (optional, requires `brotli` or `brotlicffi`
package).  Only responses larger than 1 KB are compressed — smaller payloads
are returned as-is to avoid the CPU overhead being larger than the wire saving.
"""

from __future__ import annotations

import gzip
import threading
from typing import Any, Final, Protocol, cast

from loguru import logger


class _BrotliModule(Protocol):
    """Static shape for the optional `brotli`/`brotlicffi` module.

    Neither package ships type stubs, so basedpyright can't infer anything
    about the module beyond "unresolved import" — this Protocol pins the one
    call this file makes.
    """

    def compress(self, data: bytes, quality: int = ...) -> bytes: ...


try:
    import brotli as _brotli  # type: ignore[import-untyped]
except ImportError:
    _brotli = None

_BROTLI_AVAILABLE: Final[bool] = _brotli is not None

_MIN_COMPRESS_BYTES: Final[int] = 1024


def compress_response(body: bytes, accept_encoding: str) -> tuple[bytes, str]:
    """Compress *body* using the best algorithm the client advertises.

    Algorithm selection priority: brotli > gzip > identity.
    Only compresses when ``len(body) > 1024``.

    Returns:
        (compressed_body, content_encoding)  — encoding is ``"identity"``
        when no compression is applied.
    """
    if len(body) <= _MIN_COMPRESS_BYTES:
        return body, "identity"

    if _BROTLI_AVAILABLE and "br" in accept_encoding and _brotli is not None:
        try:
            brotli_mod = cast(_BrotliModule, cast(Any, _brotli))
            compressed = brotli_mod.compress(body, quality=4)
            return compressed, "br"
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[compression] brotli failed, falling back to gzip: {exc}")

    if "gzip" in accept_encoding:
        compressed = gzip.compress(body, compresslevel=6)
        return compressed, "gzip"

    return body, "identity"


class CompressionStats:
    """Thread-safe accumulator for response-compression telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_compressed: int = 0
        self.total_bytes_before: int = 0
        self.total_bytes_after: int = 0
        self._by_encoding: dict[str, int] = {}

    def record(self, before: int, after: int, encoding: str) -> None:
        """Record a single compression event.

        Args:
            before: Original body size in bytes.
            after:  Compressed body size in bytes.
            encoding: Content-Encoding applied (``"gzip"``, ``"br"``, …).
        """
        if encoding == "identity":
            return
        with self._lock:
            self.total_compressed += 1
            self.total_bytes_before += before
            self.total_bytes_after += after
            self._by_encoding[encoding] = self._by_encoding.get(encoding, 0) + 1

    def stats(self) -> dict[str, object]:
        """Return a JSON-serialisable stats snapshot."""
        with self._lock:
            bytes_saved = self.total_bytes_before - self.total_bytes_after
            ratio = (
                round(self.total_bytes_after / self.total_bytes_before, 4)
                if self.total_bytes_before > 0
                else 0.0
            )
            return {
                "requests_compressed": self.total_compressed,
                "bytes_before": self.total_bytes_before,
                "bytes_after": self.total_bytes_after,
                "bytes_saved": bytes_saved,
                "compression_ratio": ratio,
                "by_encoding": dict(self._by_encoding),
                "brotli_available": _BROTLI_AVAILABLE,
            }


COMPRESSION_STATS: Final[CompressionStats] = CompressionStats()
