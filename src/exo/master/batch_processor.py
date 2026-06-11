"""Batch inference processor for non-streaming bulk completions."""

import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Literal
from uuid import uuid4

from loguru import logger


@dataclass
class BatchItem:
    item_id: str
    messages: list[dict[str, Any]]
    model: str
    temperature: float
    max_tokens: int | None
    seed: int | None
    status: Literal["pending", "running", "done", "failed"] = "pending"
    result: str | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class BatchJob:
    job_id: str
    items: list[BatchItem]
    created_at: float
    completed: int = 0
    failed: int = 0
    status: Literal["pending", "running", "done", "cancelled"] = "pending"


InferenceFn = Callable[
    [list[dict[str, Any]], str, float, int | None, int | None],
    Coroutine[Any, Any, str],
]


class BatchProcessor:
    def __init__(self) -> None:
        self._jobs: dict[str, BatchJob] = {}

    def create_job(self, items: list[dict[str, Any]]) -> BatchJob:
        if not items:
            raise ValueError("items must not be empty")
        job_id = str(uuid4())
        batch_items: list[BatchItem] = []
        for raw in items:
            item_id: str = raw.get("custom_id") or str(uuid4())
            messages: list[dict[str, Any]] = raw.get("messages", [])
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"item {item_id!r}: messages must be a non-empty list")
            model: str = raw.get("model", "")
            if not model:
                raise ValueError(f"item {item_id!r}: model is required")
            temperature: float = float(raw.get("temperature") or 0.0)
            max_tokens: int | None = raw.get("max_tokens")
            if max_tokens is not None:
                max_tokens = int(max_tokens)
            seed: int | None = raw.get("seed")
            if seed is not None:
                seed = int(seed)
            batch_items.append(
                BatchItem(
                    item_id=item_id,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    seed=seed,
                )
            )
        job = BatchJob(
            job_id=job_id,
            items=batch_items,
            created_at=time.time(),
        )
        self._jobs[job_id] = job
        logger.info(f"[batch] created job_id={job_id} items={len(batch_items)}")
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job.status in ("done", "cancelled"):
            return False
        job.status = "cancelled"
        # Mark any still-pending items as failed
        for item in job.items:
            if item.status == "pending":
                item.status = "failed"
                item.error = "job cancelled"
                job.failed += 1
        logger.info(f"[batch] cancelled job_id={job_id}")
        return True

    def list_jobs(self, limit: int = 20) -> list[BatchJob]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    async def run_job(self, job_id: str, inference_fn: InferenceFn) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            logger.warning(f"[batch] run_job: unknown job_id={job_id}")
            return
        if job.status == "cancelled":
            return
        job.status = "running"
        logger.info(f"[batch] starting job_id={job_id} items={len(job.items)}")
        for item in job.items:
            if job.status == "cancelled":
                break
            if item.status != "pending":
                continue
            item.status = "running"
            item.started_at = time.time()
            try:
                result = await inference_fn(
                    item.messages,
                    item.model,
                    item.temperature,
                    item.max_tokens,
                    item.seed,
                )
                item.result = result
                item.status = "done"
                job.completed += 1
                logger.debug(f"[batch] job_id={job_id} item_id={item.item_id} done")
            except Exception as exc:
                item.status = "failed"
                item.error = str(exc)
                job.failed += 1
                logger.warning(f"[batch] job_id={job_id} item_id={item.item_id} failed: {exc}")
            finally:
                item.completed_at = time.time()
        if job.status != "cancelled":
            job.status = "done"
        logger.info(
            f"[batch] finished job_id={job_id} completed={job.completed} failed={job.failed}"
        )

    def cleanup_old_jobs(self, max_age_seconds: float = 3600.0) -> int:
        cutoff = time.time() - max_age_seconds
        to_delete = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in ("done", "cancelled") and job.created_at < cutoff
        ]
        for job_id in to_delete:
            del self._jobs[job_id]
        if to_delete:
            logger.info(f"[batch] cleanup removed {len(to_delete)} old jobs")
        return len(to_delete)


BATCH_PROCESSOR = BatchProcessor()
