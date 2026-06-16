"""node_registry.py — production-grade node capability registry for exo.

Each node advertises its RAM, compute capability, and loaded model shards.
The master uses this for intelligent placement decisions.
"""

from __future__ import annotations

import platform
import socket
import subprocess
import time
from dataclasses import dataclass
from threading import Lock
from typing import Final, Literal

import psutil
from loguru import logger

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

ComputeType = Literal["apple_silicon", "cuda", "cpu"]


@dataclass
class NodeCapability:
    node_id: str
    hostname: str
    ram_total_gb: float
    ram_available_gb: float
    compute_type: ComputeType
    compute_flops_tflops: float  # e.g. M1 Max ≈ 10.4, M4 ≈ 4.6
    tb4_link: bool  # True if a Thunderbolt 4 interface is present
    loaded_models: list[str]
    advertised_at: float  # unix timestamp (time.time())


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class NodeRegistry:
    """Thread-safe in-memory store of per-node capabilities."""

    def __init__(self) -> None:
        self._capabilities: dict[str, NodeCapability] = {}
        self._lock: Lock = Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def advertise(self, cap: NodeCapability) -> None:
        """Register or update a node's capability record."""
        with self._lock:
            self._capabilities[cap.node_id] = cap
        logger.debug(
            f"[node_registry] advertised node={cap.node_id!r} "
            f"ram_avail={cap.ram_available_gb:.1f}GB "
            f"compute={cap.compute_type} "
            f"flops={cap.compute_flops_tflops:.1f}T"
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> NodeCapability | None:
        with self._lock:
            return self._capabilities.get(node_id)

    def all_nodes(self) -> list[NodeCapability]:
        with self._lock:
            return list(self._capabilities.values())

    def best_for_model(self, model_size_gb: float) -> NodeCapability | None:
        """Return the node with the most available RAM that can fit the model."""
        with self._lock:
            candidates = [
                cap
                for cap in self._capabilities.values()
                if cap.ram_available_gb >= model_size_gb
            ]
        if not candidates:
            logger.warning(
                f"[node_registry] no node has ≥{model_size_gb:.1f}GB available "
                f"for model placement"
            )
            return None
        return max(candidates, key=lambda c: c.ram_available_gb)

    def stale_nodes(self, max_age_seconds: float = 30.0) -> list[str]:
        """Return node_ids whose last advertisement is older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        with self._lock:
            return [
                node_id
                for node_id, cap in self._capabilities.items()
                if cap.advertised_at < cutoff
            ]

    def evict_stale(self, max_age_seconds: float = 30.0) -> int:
        """Remove stale entries; returns count of evicted nodes."""
        stale = self.stale_nodes(max_age_seconds)
        if not stale:
            return 0
        with self._lock:
            for node_id in stale:
                self._capabilities.pop(node_id, None)
        logger.info(f"[node_registry] evicted {len(stale)} stale node(s): {stale}")
        return len(stale)

    # ------------------------------------------------------------------
    # Aggregate stats
    # ------------------------------------------------------------------

    def total_cluster_ram_gb(self) -> float:
        with self._lock:
            return sum(cap.ram_total_gb for cap in self._capabilities.values())

    def total_cluster_flops_tflops(self) -> float:
        with self._lock:
            return sum(cap.compute_flops_tflops for cap in self._capabilities.values())

    def stats(self) -> dict[str, object]:
        """Return a serialisable summary of the registry — used by the API."""
        nodes = self.all_nodes()
        return {
            "node_count": len(nodes),
            "total_ram_gb": round(self.total_cluster_ram_gb(), 2),
            "total_flops_tflops": round(self.total_cluster_flops_tflops(), 2),
            "nodes": [
                {
                    "node_id": cap.node_id,
                    "hostname": cap.hostname,
                    "ram_total_gb": round(cap.ram_total_gb, 2),
                    "ram_available_gb": round(cap.ram_available_gb, 2),
                    "compute_type": cap.compute_type,
                    "compute_flops_tflops": cap.compute_flops_tflops,
                    "tb4_link": cap.tb4_link,
                    "loaded_models": cap.loaded_models,
                    "advertised_at": cap.advertised_at,
                    "age_seconds": round(time.time() - cap.advertised_at, 1),
                }
                for cap in nodes
            ],
        }


# ---------------------------------------------------------------------------
# Local capability detection
# ---------------------------------------------------------------------------

# Mapping of Apple Silicon brand-string fragments → GPU TFLOPS (FP16).
# Values are approximate published figures.
_APPLE_SILICON_FLOPS: Final[dict[str, float]] = {
    "M1 Max": 10.4,
    "M1 Ultra": 21.2,
    "M1 Pro": 5.2,
    "M1": 2.6,
    "M2 Max": 13.6,
    "M2 Ultra": 27.2,
    "M2 Pro": 6.8,
    "M2": 3.6,
    "M3 Max": 14.2,
    "M3 Ultra": 28.4,
    "M3 Pro": 7.0,
    "M3": 3.6,
    "M4 Max": 14.2,
    "M4 Ultra": 28.4,
    "M4 Pro": 9.2,
    "M4": 4.6,
}

_DEFAULT_APPLE_SILICON_FLOPS: Final[float] = 4.0  # conservative fallback
_DEFAULT_CUDA_FLOPS: Final[float] = 10.0  # generic GPU placeholder
_DEFAULT_CPU_FLOPS: Final[float] = 0.5  # typical AVX2 system


def _sysctl(key: str) -> str:
    """Read a macOS sysctl value; returns empty string on failure."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _detect_compute_type() -> ComputeType:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "apple_silicon"
    try:
        import importlib.util

        if importlib.util.find_spec("torch") is not None:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                return "cuda"
    except Exception as exc:
        logger.debug(f"[node_registry] accelerator probe failed: {exc}")
    return "cpu"


def _detect_flops(compute_type: ComputeType) -> float:
    if compute_type == "apple_silicon":
        brand = _sysctl("machdep.cpu.brand_string")
        if not brand:
            # arm64 Macs may expose chip info differently
            brand = _sysctl("hw.model")
        for fragment, flops in _APPLE_SILICON_FLOPS.items():
            if fragment in brand:
                logger.debug(
                    f"[node_registry] detected chip fragment={fragment!r} "
                    f"→ {flops}T FLOPS"
                )
                return flops
        logger.debug(
            f"[node_registry] unknown Apple Silicon brand={brand!r}, "
            f"using default {_DEFAULT_APPLE_SILICON_FLOPS}T"
        )
        return _DEFAULT_APPLE_SILICON_FLOPS

    if compute_type == "cuda":
        try:
            import torch  # type: ignore[import-untyped]

            props = torch.cuda.get_device_properties(0)
            # Rough FP16 estimate from SM count × 2 × clock
            sm_count: int = props.multi_processor_count
            clock_ghz: float = props.clock_rate / 1e6  # clock_rate is in kHz
            cores_per_sm: int = 128  # conservative for Ampere/Ada
            flops = (sm_count * cores_per_sm * 2 * clock_ghz) / 1000  # TFLOPS
            return round(flops, 1)
        except Exception:
            return _DEFAULT_CUDA_FLOPS

    return _DEFAULT_CPU_FLOPS


def _detect_ram_gb() -> tuple[float, float]:
    """Return (total_gb, available_gb)."""
    if platform.system() == "Darwin":
        raw = _sysctl("hw.memsize")
        if raw:
            try:
                total_bytes = int(raw)
                total_gb = total_bytes / (1024**3)
                vm = psutil.virtual_memory()
                available_gb = vm.available / (1024**3)
                return round(total_gb, 2), round(available_gb, 2)
            except ValueError as exc:
                logger.debug(f"[node_registry] sysctl memsize unparseable: {exc}")
    vm = psutil.virtual_memory()
    return (
        round(vm.total / (1024**3), 2),
        round(vm.available / (1024**3), 2),
    )


def _detect_tb4() -> bool:
    """
    Heuristic: on macOS arm64 assume TB4 present (all post-2020 Apple Silicon
    MacBooks ship with at least one USB4/TB4 port). On other platforms check
    for a 'thunderbolt' network interface name (rare but possible on Linux).
    """
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return True
    import psutil as _psutil

    interfaces = list(_psutil.net_if_addrs().keys())
    return any("thunderbolt" in iface.lower() for iface in interfaces)


def local_capability(node_id: str) -> NodeCapability:
    """Auto-detect and return a NodeCapability for the current machine."""
    compute_type = _detect_compute_type()
    total_gb, available_gb = _detect_ram_gb()
    flops = _detect_flops(compute_type)
    tb4 = _detect_tb4()
    hostname = socket.gethostname()

    cap = NodeCapability(
        node_id=node_id,
        hostname=hostname,
        ram_total_gb=total_gb,
        ram_available_gb=available_gb,
        compute_type=compute_type,
        compute_flops_tflops=flops,
        tb4_link=tb4,
        loaded_models=[],
        advertised_at=time.time(),
    )
    logger.info(
        f"[node_registry] local capability: node={node_id!r} host={hostname!r} "
        f"ram={total_gb:.1f}GB total / {available_gb:.1f}GB free "
        f"compute={compute_type} flops={flops}T tb4={tb4}"
    )
    return cap


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

NODE_REGISTRY: Final[NodeRegistry] = NodeRegistry()
