"""
Webhook notifier: sends HTTP POST to registered webhooks on key cluster events
(worker_down, model_loaded, cluster_panic, quota_exceeded, slo_violated).
Webhooks registered via API or EXO_WEBHOOK_URL env var.
Uses httpx with 5s timeout; failures are logged but do not affect hot path.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

try:
    import httpx as _httpx
    _HAS_HTTPX = True
except ImportError:
    _httpx = None  # type: ignore[assignment]
    _HAS_HTTPX = False
    logger.debug("httpx not available — webhook notifications disabled")

_EVENT_TYPES = frozenset({
    "worker_down", "worker_up", "model_loaded", "cluster_panic",
    "quota_exceeded", "slo_violated", "split_brain", "quorum_lost",
    "election", "model_downloaded",
})


def _make_webhook_id(url: str, events: set[str]) -> str:
    """Deterministic ID: sha256(url + sorted_events)[:16]."""
    key = url + "|" + ",".join(sorted(events))
    return hashlib.sha256(key.encode()).hexdigest()[:16]


@dataclass
class WebhookRegistration:
    url: str
    webhook_id: str = ""
    secret: str = ""   # HMAC-SHA256 secret for signature
    events: set[str] = field(default_factory=lambda: set(_EVENT_TYPES))
    created_at: float = field(default_factory=time.time)
    delivery_count: int = 0
    failure_count: int = 0

    def __post_init__(self) -> None:
        if not self.webhook_id:
            self.webhook_id = _make_webhook_id(self.url, self.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "webhook_id": self.webhook_id,
            "url": self.url[:50] + "…" if len(self.url) > 50 else self.url,
            "events": sorted(self.events),
            "delivery_count": self.delivery_count,
            "failure_count": self.failure_count,
        }


_DEFAULT_MAX_RETRIES = 3


class WebhookNotifier:
    """
    register(url, secret, events): adds a webhook endpoint (deduped by url+events).
    unregister(webhook_id): removes a webhook by its deterministic ID.
    notify(event_type, payload): fire-and-forget; schedules delivery without blocking.
    list(): returns registered WebhookRegistration objects.
    webhook_count: number of registered webhooks.
    stats(): aggregate delivery/failure counters.
    """

    def __init__(self, max_retries: int = _DEFAULT_MAX_RETRIES) -> None:
        self.max_retries = max_retries
        self._webhooks: dict[str, WebhookRegistration] = {}  # webhook_id → registration
        self._total_fired: int = 0
        self._total_failed: int = 0
        self._load_env()

    def _load_env(self) -> None:
        url = os.getenv("EXO_WEBHOOK_URL", "")
        if url:
            self.register(url, secret=os.getenv("EXO_WEBHOOK_SECRET", ""))
            logger.info("WebhookNotifier: loaded URL from EXO_WEBHOOK_URL")

    def register(self, url: str, secret: str = "", events: set[str] | None = None) -> str:
        """Register a webhook; silently dedup by (url, events). Returns webhook_id."""
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"WebhookNotifier: url must start with http:// or https://, got {url[:40]!r}")
        ev = events if events is not None else set(_EVENT_TYPES)
        wh = WebhookRegistration(url=url, secret=secret, events=ev)
        if wh.webhook_id not in self._webhooks:
            self._webhooks[wh.webhook_id] = wh
            logger.info(f"WebhookNotifier: registered id={wh.webhook_id} url={url[:40]}")
        return wh.webhook_id

    def unregister(self, webhook_id: str) -> bool:
        """Remove webhook by ID. Returns True if it existed."""
        existed = webhook_id in self._webhooks
        self._webhooks.pop(webhook_id, None)
        return existed

    @property
    def webhook_count(self) -> int:
        return len(self._webhooks)

    def list(self) -> list[WebhookRegistration]:
        return list(self._webhooks.values())

    def notify(self, event_type: str, payload: dict[str, Any]) -> None:
        """Fire-and-forget: schedules HTTP delivery without blocking the caller."""
        if not _HAS_HTTPX or not self._webhooks:
            return
        try:
            loop = asyncio.get_running_loop()
            # A running event loop exists — schedule as a fire-and-forget task.
            loop.create_task(self._dispatch(event_type, payload))
        except RuntimeError:
            # No running event loop (e.g. called from a sync context or test).
            # Run synchronously on a fresh loop.
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._dispatch(event_type, payload))
            finally:
                loop.close()

    async def _dispatch(self, event_type: str, payload: dict[str, Any]) -> None:
        body = json.dumps({"event": event_type, "timestamp": time.time(), "data": payload})

        async def _send(wh: WebhookRegistration) -> None:
            if event_type not in wh.events:
                return
            headers: dict[str, str] = {"Content-Type": "application/json"}
            if wh.secret:
                sig = hmac.new(wh.secret.encode(), body.encode(), hashlib.sha256).hexdigest()
                headers["X-Exo-Signature"] = f"sha256={sig}"
            last_exc: Exception | None = None
            for attempt in range(self.max_retries):
                try:
                    async with _httpx.AsyncClient(timeout=5.0) as client:
                        resp = await client.post(wh.url, content=body, headers=headers)
                    wh.delivery_count += 1
                    self._total_fired += 1
                    if resp.status_code >= 400:
                        wh.failure_count += 1
                        self._total_failed += 1
                        logger.warning(f"Webhook: {wh.url[:40]} HTTP {resp.status_code} (attempt {attempt+1})")
                    return
                except Exception as exc:
                    last_exc = exc
                    logger.debug(f"Webhook: send failed {wh.url[:40]} attempt={attempt+1}: {exc}")
            if last_exc is not None:
                wh.failure_count += 1
                self._total_failed += 1

        await asyncio.gather(*[_send(wh) for wh in self._webhooks.values()], return_exceptions=True)

    def stats(self) -> dict[str, Any]:
        return {
            "total_fired": self._total_fired,
            "total_failed": self._total_failed,
            "registered": len(self._webhooks),
            "webhooks": [wh.to_dict() for wh in self._webhooks.values()],
        }

    def get_stats(self) -> dict[str, Any]:
        return self.stats()


WEBHOOK_NOTIFIER = WebhookNotifier()
