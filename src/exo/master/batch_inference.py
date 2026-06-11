from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


_BATCH_TIMEOUT = 60.0   # seconds per item
_MAX_BATCH_SIZE = 32


@dataclass
class BatchItem:
    index: int
    messages: list[dict[str, Any]]
    model: str
    max_tokens: int = 512
    temperature: float = 1.0


@dataclass
class BatchResult:
    index: int
    content: str | None
    error: str | None
    latency_ms: float
    tokens_generated: int = 0


class BatchInferenceHandler:
    """
    Runs multiple chat completion requests in parallel and collects results.
    Each item is dispatched concurrently; failures are isolated per item.
    """

    def __init__(self) -> None:
        self._total_batches: int = 0
        self._total_items: int = 0
        self._total_latency_ms: float = 0.0

    def stats(self) -> dict[str, Any]:
        avg = (self._total_latency_ms / self._total_batches) if self._total_batches > 0 else 0.0
        avg_batch_size = (self._total_items / self._total_batches) if self._total_batches > 0 else 0.0
        return {
            "total_batches": self._total_batches,
            "total_items": self._total_items,
            "avg_batch_size": round(avg_batch_size, 2),
            "avg_latency_ms": round(avg, 1),
        }

    async def run_batch(
        self,
        items: list[BatchItem],
        inference_fn: Any,  # async callable(messages, model, max_tokens, temperature) -> str
    ) -> list[BatchResult]:
        if len(items) > _MAX_BATCH_SIZE:
            raise ValueError(f"Batch size {len(items)} exceeds max {_MAX_BATCH_SIZE}")

        async def _run_one(item: BatchItem) -> BatchResult:
            start = time.monotonic()
            try:
                content = await asyncio.wait_for(
                    inference_fn(item.messages, item.model, item.max_tokens, item.temperature),
                    timeout=_BATCH_TIMEOUT,
                )
                return BatchResult(
                    index=item.index,
                    content=content,
                    error=None,
                    latency_ms=(time.monotonic() - start) * 1000,
                    tokens_generated=len(content.split()) if content else 0,
                )
            except asyncio.TimeoutError:
                logger.warning(f"BatchInference item={item.index} timeout after {_BATCH_TIMEOUT}s")
                return BatchResult(
                    index=item.index,
                    content=None,
                    error="timeout",
                    latency_ms=(time.monotonic() - start) * 1000,
                )
            except Exception as exc:
                logger.warning(f"BatchInference item={item.index} error: {exc}")
                return BatchResult(
                    index=item.index,
                    content=None,
                    error=str(exc),
                    latency_ms=(time.monotonic() - start) * 1000,
                )

        results = await asyncio.gather(*[_run_one(item) for item in items])
        results = sorted(results, key=lambda r: r.index)
        self._total_batches += 1
        self._total_items += len(results)
        self._total_latency_ms += sum(r.latency_ms for r in results)
        return results

    def format_response(self, results: list[BatchResult]) -> dict[str, Any]:
        return {
            "results": [
                {
                    "index": r.index,
                    "content": r.content,
                    "error": r.error,
                    "latency_ms": round(r.latency_ms, 1),
                    "tokens_generated": r.tokens_generated,
                }
                for r in results
            ],
            "total": len(results),
            "errors": sum(1 for r in results if r.error),
        }


BATCH_INFERENCE = BatchInferenceHandler()
