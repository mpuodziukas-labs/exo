"""
API response caching headers: sets correct HTTP cache-control headers on all
API responses. Inference responses are never cached (private, no-store).
Static/info endpoints get short max-age. Health endpoints get no-cache with revalidation.
"""

from __future__ import annotations

from typing import Any

# Endpoint prefix → cache policy
_POLICIES: dict[str, dict[str, str]] = {
    # Inference — never cache
    "/v1/chat/completions": {"Cache-Control": "no-store, private"},
    "/v1/completions": {"Cache-Control": "no-store, private"},
    # Health — allow revalidation but no stale serving
    "/v1/health": {"Cache-Control": "no-cache"},
    "/v1/metrics": {"Cache-Control": "no-cache"},
    # Mostly-static info endpoints — short TTL
    "/v1/models": {"Cache-Control": "public, max-age=30"},
    "/v1/deps": {"Cache-Control": "public, max-age=60"},
    "/v1/singletons": {"Cache-Control": "public, max-age=60"},
    # Snapshot data — can be cached longer
    "/v1/snapshots": {"Cache-Control": "public, max-age=120"},
}

_DEFAULT_POLICY = {"Cache-Control": "no-cache"}


def get_cache_headers(path: str) -> dict[str, str]:
    """Return appropriate Cache-Control headers for a given request path."""
    for prefix, headers in _POLICIES.items():
        if path.startswith(prefix):
            return headers
    return _DEFAULT_POLICY


class CacheHeaderMiddlewareHelper:
    """
    apply(path, response_headers): mutates response_headers dict with cache policy.
    get_policy(path): returns the policy dict for observability.
    """

    def apply(self, path: str, response_headers: dict[str, str]) -> None:
        headers = get_cache_headers(path)
        response_headers.update(headers)

    def get_policy(self, path: str) -> dict[str, Any]:
        return {"path": path, "headers": get_cache_headers(path)}

    def get_all_policies(self) -> dict[str, Any]:
        return {
            "policies": [
                {"prefix": prefix, "Cache-Control": hdrs["Cache-Control"]}
                for prefix, hdrs in _POLICIES.items()
            ],
            "default": _DEFAULT_POLICY,
        }


CACHE_HEADERS = CacheHeaderMiddlewareHelper()
