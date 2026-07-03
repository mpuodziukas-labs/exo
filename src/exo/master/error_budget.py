"""
Per-endpoint error budget tracker: tracks success/error counts per API endpoint.
Computes error rate and remaining error budget (SLO target: 99.9% success = 0.1% error budget).
Alerts when budget is >50% consumed.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from loguru import logger

_SLO_TARGET = 0.999  # 99.9% success rate
_WINDOW_SECONDS = 3600.0  # 1-hour rolling window
_ALERT_THRESHOLD = 0.5  # alert when 50% of error budget consumed


@dataclass
class RequestSample:
    timestamp: float
    success: bool


class EndpointErrorBudget:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint
        self._samples: deque[RequestSample] = deque()
        self._budget_alert_sent = False

    @property
    def budget_alert_sent(self) -> bool:
        """Whether the >50%-consumed alert is currently latched for this endpoint."""
        return self._budget_alert_sent

    def _trim(self) -> None:
        cutoff = time.time() - _WINDOW_SECONDS
        while self._samples and self._samples[0].timestamp < cutoff:
            self._samples.popleft()

    def record(self, success: bool) -> None:
        self._trim()
        self._samples.append(RequestSample(timestamp=time.time(), success=success))
        self._check_alert()

    def _check_alert(self) -> None:
        total = len(self._samples)
        if total < 10:
            return
        errors = sum(1 for s in self._samples if not s.success)
        error_rate = errors / total
        budget_consumed = error_rate / (1.0 - _SLO_TARGET)
        if budget_consumed > _ALERT_THRESHOLD and not self._budget_alert_sent:
            logger.warning(
                f"ErrorBudget: endpoint={self.endpoint} budget_consumed={budget_consumed:.1%} "
                f"error_rate={error_rate:.4%} ({errors}/{total})"
            )
            self._budget_alert_sent = True
        elif budget_consumed <= _ALERT_THRESHOLD:
            self._budget_alert_sent = False

    def to_dict(self) -> dict[str, Any]:
        self._trim()
        total = len(self._samples)
        errors = sum(1 for s in self._samples if not s.success)
        error_rate = (errors / total) if total else 0.0
        budget_consumed = error_rate / (1.0 - _SLO_TARGET)
        return {
            "endpoint": self.endpoint,
            "total_requests": total,
            "errors": errors,
            "error_rate_pct": round(error_rate * 100, 4),
            "budget_consumed_pct": round(min(budget_consumed * 100, 100.0), 2),
            "slo_met": error_rate <= (1.0 - _SLO_TARGET),
        }


class ErrorBudgetManager:
    def __init__(self) -> None:
        self._endpoints: dict[str, EndpointErrorBudget] = {}

    def record(self, endpoint: str, success: bool) -> None:
        if endpoint not in self._endpoints:
            self._endpoints[endpoint] = EndpointErrorBudget(endpoint)
        self._endpoints[endpoint].record(success)

    def get_all(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self._endpoints.values()]

    def get_summary(self) -> dict[str, Any]:
        all_budgets = self.get_all()
        violated = [b for b in all_budgets if not b["slo_met"]]
        return {
            "endpoints": len(all_budgets),
            "slo_violations": len(violated),
            "budgets": all_budgets,
        }


ERROR_BUDGET = ErrorBudgetManager()
