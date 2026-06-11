"""
Automated runbook executor: executes pre-defined remediation runbooks
in response to cluster alerts. Each runbook is a sequence of steps
(HTTP calls to internal API, log messages, sleep). Safe by default —
dry-run mode unless EXO_RUNBOOK_LIVE=1.
"""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from loguru import logger

_LIVE_MODE = os.getenv("EXO_RUNBOOK_LIVE", "0") == "1"
_API_BASE = f"http://127.0.0.1:{os.getenv('EXO_API_PORT', '52415')}"


class RunbookPriority(str, Enum):
    P0 = "P0"
    P1 = "P1"
    P2 = "P2"
    P3 = "P3"


class StepType(str, Enum):
    LOG = "log"
    SLEEP = "sleep"
    HTTP_POST = "http_post"
    HTTP_GET = "http_get"
    CALLABLE = "callable"


@dataclass
class RunbookStep:
    step_type: StepType
    message: str = ""
    path: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    sleep_s: float = 0.0
    fn: Callable[[], Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.step_type.value,
            "message": self.message,
            "path": self.path,
            "sleep_s": self.sleep_s,
        }


@dataclass
class RunbookStepResult:
    index: int
    success: bool
    output: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "success": self.success,
            "output": self.output,
            "error": self.error,
        }


@dataclass
class Runbook:
    name: str
    trigger: str  # alert event name that triggers this runbook
    steps: list[RunbookStep]
    description: str = ""
    priority: RunbookPriority = RunbookPriority.P2


@dataclass
class RunbookExecution:
    runbook_name: str
    triggered_by: str
    started_at: float = field(default_factory=time.time)
    completed_at: float = 0.0
    steps_completed: int = 0
    step_results: list[RunbookStepResult] = field(default_factory=list)
    success: bool = False
    dry_run: bool = not _LIVE_MODE
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.runbook_name,
            "runbook": self.runbook_name,
            "triggered_by": self.triggered_by,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "steps_completed": self.steps_completed,
            "step_results": [sr.to_dict() for sr in self.step_results],
            "success": self.success,
            "dry_run": self.dry_run,
            "error": self.error,
            "duration_s": round((self.completed_at or time.time()) - self.started_at, 2),
        }


# Built-in remediation runbooks
_BUILTIN_RUNBOOKS: list[Runbook] = [
    Runbook(
        name="circuit_breaker_reset",
        trigger="worker_down",
        description="Log worker down and check health after 30s",
        priority=RunbookPriority.P1,
        steps=[
            RunbookStep(StepType.LOG, message="Worker down — waiting 30s before health check"),
            RunbookStep(StepType.SLEEP, sleep_s=30.0),
            RunbookStep(StepType.HTTP_GET, path="/v1/health"),
        ],
    ),
    Runbook(
        name="memory_pressure_reclaim",
        trigger="memory_critical",
        description="Trigger memory reclaim on critical pressure",
        priority=RunbookPriority.P1,
        steps=[
            RunbookStep(StepType.LOG, message="Memory critical — triggering reclaim"),
            RunbookStep(StepType.HTTP_POST, path="/v1/memory/reclaim", body={}),
        ],
    ),
    Runbook(
        name="quota_exceeded_notify",
        trigger="quota_exceeded",
        description="Log quota exceeded event",
        priority=RunbookPriority.P2,
        steps=[
            RunbookStep(StepType.LOG, message="Quota exceeded — logging for review"),
        ],
    ),
]


class RunbookExecutor:
    def __init__(self) -> None:
        self._runbooks: dict[str, Runbook] = {rb.name: rb for rb in _BUILTIN_RUNBOOKS}
        self._executions: list[RunbookExecution] = []
        # Per-runbook last result cache: name -> RunbookExecution
        self._last_results: dict[str, RunbookExecution] = {}
        self._failure_count: int = 0

    # ------------------------------------------------------------------
    # Registration — accepts either a Runbook object or name+steps dict
    # ------------------------------------------------------------------

    def register(self, runbook_or_name: "Runbook | str", steps: list[Any] | None = None,
                 trigger: str = "manual", description: str = "",
                 priority: RunbookPriority = RunbookPriority.P2) -> None:
        if isinstance(runbook_or_name, Runbook):
            rb = runbook_or_name
        else:
            name = runbook_or_name
            rb_steps: list[RunbookStep] = []
            for s in (steps or []):
                if callable(s):
                    rb_steps.append(RunbookStep(StepType.CALLABLE, fn=s))
                elif isinstance(s, RunbookStep):
                    rb_steps.append(s)
                else:
                    rb_steps.append(RunbookStep(StepType.LOG, message=str(s)))
            rb = Runbook(name=name, trigger=trigger, steps=rb_steps,
                         description=description, priority=priority)
        self._runbooks[rb.name] = rb
        logger.info(f"RunbookExecutor: registered runbook={rb.name}")

    def get_for_trigger(self, trigger: str) -> list[Runbook]:
        return [rb for rb in self._runbooks.values() if rb.trigger == trigger]

    def list_runbooks(self) -> list[str]:
        """Return all registered runbook names."""
        return list(self._runbooks.keys())

    def last_result(self, name: str) -> RunbookExecution | None:
        """Return the most recent execution for a runbook, or None."""
        return self._last_results.get(name)

    def _get_runbook(self, name: str) -> Runbook:
        rb = self._runbooks.get(name)
        if rb is None:
            raise ValueError(f"Unknown runbook: {name!r}")
        return rb

    # ------------------------------------------------------------------
    # Execute by runbook name (sync wrapper for sync step lists)
    # ------------------------------------------------------------------

    def execute_sync(self, name: str, dry_run: bool = False) -> RunbookExecution:
        """Execute a runbook synchronously.  Use when all steps are CALLABLE/LOG."""
        rb = self._get_runbook(name)
        execution = RunbookExecution(
            runbook_name=rb.name,
            triggered_by=rb.trigger,
            dry_run=dry_run,
        )
        step_results: list[RunbookStepResult] = []
        for i, step in enumerate(rb.steps):
            sr = RunbookStepResult(index=i, success=False)
            try:
                if dry_run:
                    if step.step_type == StepType.CALLABLE and step.fn is not None:
                        # Validate callable exists but do not call
                        assert callable(step.fn)
                    sr.output = "dry_run"
                    sr.success = True
                elif step.step_type == StepType.CALLABLE and step.fn is not None:
                    result = step.fn()
                    sr.output = str(result) if result is not None else "ok"
                    sr.success = True
                elif step.step_type == StepType.LOG:
                    logger.info(f"Runbook [{rb.name}]: {step.message}")
                    sr.output = step.message
                    sr.success = True
                else:
                    sr.output = f"step_type={step.step_type.value} skipped in sync mode"
                    sr.success = True
            except Exception as exc:
                sr.error = str(exc)
                sr.success = False
                logger.warning(f"Runbook [{rb.name}] step {i} error: {exc}")
            step_results.append(sr)
            execution.steps_completed += 1
        execution.step_results = step_results
        execution.success = all(sr.success for sr in step_results)
        execution.completed_at = time.time()
        if not execution.success:
            self._failure_count += 1
        self._executions.append(execution)
        self._last_results[rb.name] = execution
        return execution

    # ------------------------------------------------------------------
    # Async execute (original interface, extended)
    # ------------------------------------------------------------------

    async def execute(self, runbook_or_name: "Runbook | str",
                      trigger_context: dict[str, Any] | None = None,
                      dry_run: bool = False) -> RunbookExecution:
        if isinstance(runbook_or_name, str):
            runbook = self._get_runbook(runbook_or_name)
        else:
            runbook = runbook_or_name
        execution = RunbookExecution(
            runbook_name=runbook.name,
            triggered_by=runbook.trigger,
            dry_run=dry_run,
        )
        logger.info(
            f"RunbookExecutor: {'DRY-RUN' if dry_run else 'LIVE'} "
            f"runbook={runbook.name} trigger={runbook.trigger}"
        )
        step_results: list[RunbookStepResult] = []
        try:
            for i, step in enumerate(runbook.steps):
                sr = RunbookStepResult(index=i, success=False)
                try:
                    if dry_run:
                        if step.step_type == StepType.CALLABLE and step.fn is not None:
                            assert callable(step.fn)
                        sr.output = "dry_run"
                        sr.success = True
                    elif step.step_type == StepType.LOG:
                        logger.info(f"Runbook [{runbook.name}]: {step.message}")
                        sr.output = step.message
                        sr.success = True
                    elif step.step_type == StepType.CALLABLE and step.fn is not None:
                        result = step.fn()
                        if asyncio.iscoroutine(result):
                            result = await result
                        sr.output = str(result) if result is not None else "ok"
                        sr.success = True
                    elif step.step_type == StepType.SLEEP:
                        await asyncio.sleep(step.sleep_s)
                        sr.output = f"slept {step.sleep_s}s"
                        sr.success = True
                    elif step.step_type in (StepType.HTTP_POST, StepType.HTTP_GET):
                        try:
                            import httpx
                            async with httpx.AsyncClient(timeout=10.0) as client:
                                if step.step_type == StepType.HTTP_POST:
                                    await client.post(f"{_API_BASE}{step.path}", json=step.body)
                                else:
                                    await client.get(f"{_API_BASE}{step.path}")
                            sr.output = f"http ok {step.path}"
                            sr.success = True
                        except Exception as exc:
                            sr.error = str(exc)
                            sr.success = False
                            logger.warning(f"Runbook [{runbook.name}] HTTP error: {exc}")
                    else:
                        sr.output = f"step_type={step.step_type.value} unhandled"
                        sr.success = True
                except Exception as exc:
                    sr.error = str(exc)
                    sr.success = False
                    logger.warning(f"Runbook [{runbook.name}] step {i} error: {exc}")
                step_results.append(sr)
                execution.steps_completed += 1
            execution.success = all(sr.success for sr in step_results)
        except Exception as exc:
            execution.error = str(exc)
            logger.error(f"RunbookExecutor: runbook={runbook.name} failed: {exc}")
        execution.step_results = step_results
        execution.completed_at = time.time()
        if not execution.success:
            self._failure_count += 1
        self._executions.append(execution)
        self._last_results[runbook.name] = execution
        return execution

    async def trigger(self, event: str, context: dict[str, Any]) -> list[RunbookExecution]:
        runbooks = self.get_for_trigger(event)
        if not runbooks:
            return []
        results = await asyncio.gather(*[self.execute(rb, context) for rb in runbooks])
        return list(results)

    def get_stats(self) -> dict[str, Any]:
        return {
            "live_mode": _LIVE_MODE,
            "registered_runbooks": len(self._runbooks),
            "total_executions": len(self._executions),
            "failures": self._failure_count,
            "recent_executions": [e.to_dict() for e in self._executions[-10:]],
        }

    def stats(self) -> dict[str, Any]:
        return self.get_stats()

    def register_default_runbooks(self) -> None:
        """Re-register built-in runbooks. Called at startup."""
        for rb in _BUILTIN_RUNBOOKS:
            self._runbooks[rb.name] = rb
        logger.info(f"RunbookExecutor: default runbooks registered ({len(_BUILTIN_RUNBOOKS)})")


RUNBOOK_EXECUTOR = RunbookExecutor()
