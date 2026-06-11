"""
singleton_healthcheck_v2.py — Fully testable, registry-based singleton health checker.

Unlike singleton_healthcheck.py (which uses import-based discovery), this class
accepts explicit registrations of (name, check_fn) pairs. Designed for:
  - Startup self-tests
  - Unit-testable without any real singletons
  - Per-check timeout enforcement (2 s default)

Karpathy rule: one function, one job.  Fail loudly.
"""
from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

_DEFAULT_TIMEOUT_S: float = 2.0


@dataclass
class CheckEntry:
    name: str
    check_fn: Callable[[], Any]


@dataclass
class CheckResult:
    name: str
    status: str          # "ok" | "error" | "timeout"
    error: str = ""
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "duration_s": round(self.duration_s, 4),
        }


class SingletonHealthcheck:
    """
    Registry-based singleton health checker.

    Usage::

        hc = SingletonHealthcheck()
        hc.register("my_singleton", lambda: assert my_obj is not None)
        results = hc.run_all()
        print(hc.summary())
    """

    def __init__(self, timeout_s: float = _DEFAULT_TIMEOUT_S) -> None:
        self._timeout_s = timeout_s
        self._entries: dict[str, CheckEntry] = {}
        self._last_results: list[CheckResult] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, check_fn: Callable[[], Any]) -> None:
        """Register a named health check function."""
        if not callable(check_fn):
            raise TypeError(f"check_fn for {name!r} must be callable")
        self._entries[name] = CheckEntry(name=name, check_fn=check_fn)
        logger.debug(f"SingletonHealthcheck: registered check={name!r}")

    def reset(self) -> None:
        """Clear all registrations and cached results."""
        self._entries.clear()
        self._last_results.clear()
        logger.debug("SingletonHealthcheck: reset")

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run_one(self, entry: CheckEntry) -> CheckResult:
        t0 = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(entry.check_fn)
            try:
                future.result(timeout=self._timeout_s)
                duration = time.monotonic() - t0
                return CheckResult(name=entry.name, status="ok", duration_s=duration)
            except concurrent.futures.TimeoutError:
                duration = time.monotonic() - t0
                logger.warning(f"SingletonHealthcheck: timeout on check={entry.name!r}")
                return CheckResult(
                    name=entry.name, status="timeout",
                    error=f"exceeded {self._timeout_s}s timeout",
                    duration_s=duration,
                )
            except Exception as exc:
                duration = time.monotonic() - t0
                logger.warning(f"SingletonHealthcheck: error on check={entry.name!r}: {exc}")
                return CheckResult(
                    name=entry.name, status="error",
                    error=str(exc),
                    duration_s=duration,
                )

    def run_all(self) -> list[CheckResult]:
        """Run every registered check and return results."""
        results = [self._run_one(entry) for entry in self._entries.values()]
        self._last_results = results
        ok = sum(1 for r in results if r.status == "ok")
        logger.info(f"SingletonHealthcheck: {ok}/{len(results)} healthy")
        return results

    def run_subset(self, names: list[str]) -> list[CheckResult]:
        """Run only the named checks."""
        results: list[CheckResult] = []
        for name in names:
            entry = self._entries.get(name)
            if entry is None:
                raise ValueError(f"Unknown check: {name!r}")
            results.append(self._run_one(entry))
        self._last_results = results
        return results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, int]:
        """Return total/healthy/unhealthy counts from the last run_all()."""
        total = len(self._last_results)
        healthy = sum(1 for r in self._last_results if r.status == "ok")
        return {"total": total, "healthy": healthy, "unhealthy": total - healthy}

    def unhealthy_names(self) -> list[str]:
        """Return names of checks that did not pass in the last run."""
        return [r.name for r in self._last_results if r.status != "ok"]

    def is_healthy(self) -> bool:
        """True only if every check in the last run passed."""
        if not self._last_results:
            return False
        return all(r.status == "ok" for r in self._last_results)

    def get_report(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "results": [r.to_dict() for r in self._last_results],
        }


# Module-level singleton for use in prod wiring
SINGLETON_HEALTHCHECK = SingletonHealthcheck()
