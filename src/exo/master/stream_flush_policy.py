"""
Streaming token buffer flush policy: controls when buffered tokens are flushed
to the SSE stream. Supports three modes:
  - immediate: flush each token as generated (lowest latency)
  - word:      flush on word boundary (spaces/punctuation) for smoother UX
  - sentence:  flush on sentence boundary (period/newline) for lowest overhead
Configurable via EXO_FLUSH_MODE env var.
"""

from __future__ import annotations

import os
import re
from enum import Enum
from typing import Any

from loguru import logger

_FLUSH_MODE_ENV = os.getenv("EXO_FLUSH_MODE", "immediate")


class FlushMode(str, Enum):
    IMMEDIATE = "immediate"
    WORD = "word"
    SENTENCE = "sentence"


_WORD_BOUNDARY = re.compile(r"[\s,;:!?\"\'\(\)\[\]{}<>]")
_SENTENCE_BOUNDARY = re.compile(r"[.!?\n]")


class StreamFlushPolicy:
    """
    should_flush(buffer: str) → bool: returns True if the buffer should be flushed now.
    mode: current FlushMode.
    """

    def __init__(self, mode: FlushMode | None = None) -> None:
        if mode is not None:
            self._mode = mode
        else:
            try:
                self._mode = FlushMode(_FLUSH_MODE_ENV)
            except ValueError:
                logger.warning(
                    f"StreamFlushPolicy: unknown mode '{_FLUSH_MODE_ENV}', using immediate"
                )
                self._mode = FlushMode.IMMEDIATE
        logger.info(f"StreamFlushPolicy: mode={self._mode.value}")

    @property
    def mode(self) -> FlushMode:
        return self._mode

    def set_mode(self, mode: str) -> None:
        try:
            self._mode = FlushMode(mode)
            logger.info(f"StreamFlushPolicy: mode changed to {self._mode.value}")
        except ValueError:
            logger.warning(f"StreamFlushPolicy: invalid mode '{mode}'")

    def should_flush(self, buffer: str) -> bool:
        if self._mode == FlushMode.IMMEDIATE:
            return True
        if self._mode == FlushMode.WORD:
            return bool(_WORD_BOUNDARY.search(buffer))
        if self._mode == FlushMode.SENTENCE:
            return bool(_SENTENCE_BOUNDARY.search(buffer))
        return True  # default safe

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": self._mode.value,
            "available_modes": [m.value for m in FlushMode],
        }


FLUSH_POLICY = StreamFlushPolicy()
