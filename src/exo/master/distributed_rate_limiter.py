from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

_DEFAULT_RPM = 60        # requests per minute per client
_DEFAULT_TPM = 100_000   # tokens per minute per client
_WINDOW_S = 60.0


@dataclass
class ClientRateState:
    client_id: str
    rpm_limit: int = _DEFAULT_RPM
    tpm_limit: int = _DEFAULT_TPM
    # Sliding window: list of (timestamp, token_count)
    _requests: list[tuple[float, int]] = field(default_factory=list)

    def _prune(self) -> None:
        cutoff = time.monotonic() - _WINDOW_S
        self._requests = [(t, tok) for t, tok in self._requests if t >= cutoff]

    def check_and_consume(self, tokens: int) -> tuple[bool, str]:
        """Returns (allowed, reason). Allowed=False means rate limited."""
        self._prune()
        req_count = len(self._requests)
        tok_count = sum(tok for _, tok in self._requests)

        if req_count >= self.rpm_limit:
            return False, f"RPM limit {self.rpm_limit} exceeded ({req_count} in window)"
        if tok_count + tokens > self.tpm_limit:
            return False, f"TPM limit {self.tpm_limit} exceeded ({tok_count}+{tokens}>{self.tpm_limit})"

        self._requests.append((time.monotonic(), tokens))
        return True, "ok"

    def status(self) -> dict[str, Any]:
        self._prune()
        req_count = len(self._requests)
        tok_count = sum(tok for _, tok in self._requests)
        return {
            "client_id": self.client_id,
            "rpm_used": req_count,
            "rpm_limit": self.rpm_limit,
            "tpm_used": tok_count,
            "tpm_limit": self.tpm_limit,
            "rpm_remaining": max(0, self.rpm_limit - req_count),
            "tpm_remaining": max(0, self.tpm_limit - tok_count),
        }


class DistributedRateLimiter:
    """
    Cluster-wide rate limiter with per-client RPM and TPM limits.
    State is per-master in-memory; workers submit to master for check.

    Supports custom limits per client via set_limits().
    """

    def __init__(self) -> None:
        self._clients: dict[str, ClientRateState] = {}
        self._total_allowed = 0
        self._total_denied = 0

    def set_limits(self, client_id: str, rpm: int | None = None, tpm: int | None = None) -> None:
        state = self._get_or_create(client_id)
        if rpm is not None:
            state.rpm_limit = rpm
        if tpm is not None:
            state.tpm_limit = tpm
        logger.info(f"DistributedRateLimit set client={client_id} rpm={state.rpm_limit} tpm={state.tpm_limit}")

    def _get_or_create(self, client_id: str) -> ClientRateState:
        if client_id not in self._clients:
            self._clients[client_id] = ClientRateState(client_id=client_id)
        return self._clients[client_id]

    def check(self, client_id: str, tokens: int = 0) -> tuple[bool, str]:
        state = self._get_or_create(client_id)
        allowed, reason = state.check_and_consume(tokens)
        if allowed:
            self._total_allowed += 1
        else:
            self._total_denied += 1
            logger.warning(f"DistributedRateLimit DENY client={client_id}: {reason}")
        return allowed, reason

    def get_client_status(self, client_id: str) -> dict[str, Any] | None:
        state = self._clients.get(client_id)
        return state.status() if state else None

    def all_clients(self) -> list[dict[str, Any]]:
        return [s.status() for s in self._clients.values()]

    def global_stats(self) -> dict[str, Any]:
        return {
            "total_allowed": self._total_allowed,
            "total_denied": self._total_denied,
            "deny_rate": round(self._total_denied / max(self._total_allowed + self._total_denied, 1), 4),
            "client_count": len(self._clients),
        }


DISTRIBUTED_RATE_LIMITER = DistributedRateLimiter()
