from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, cast

from loguru import logger

_REGISTRY_PATH = Path.home() / ".exo" / "model_registry.json"
_lock = Lock()


def _load() -> dict[str, Any]:
    if _REGISTRY_PATH.exists():
        try:
            return cast(dict[str, Any], json.loads(_REGISTRY_PATH.read_text()))
        except Exception as exc:
            logger.warning(f"[model_registry] failed to parse registry file: {exc}")
            return {}
    return {}


def _save(data: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    _REGISTRY_PATH.write_text(json.dumps(data, indent=2))


def register_model(model_id: str, path: str | None = None) -> str:
    version_hash = hashlib.sha256(f"{model_id}:{time.time()}".encode()).hexdigest()[:12]
    with _lock:
        data = _load()
        data[model_id] = {
            "model_id": model_id,
            "version_hash": version_hash,
            "loaded_at": time.time(),
            "path": path or "",
        }
        _save(data)
    return version_hash


def get_model_info(model_id: str) -> dict[str, Any] | None:
    with _lock:
        return _load().get(model_id)


def list_models() -> list[dict[str, Any]]:
    with _lock:
        return list(_load().values())
