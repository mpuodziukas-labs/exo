from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from loguru import logger


class _CudaDeviceProps(Protocol):
    """Static shape for the fields read off torch's device-properties object.

    torch's own stub types the return of ``get_device_properties`` via a
    conditionally-assigned class (dummy fallback when built without CUDA),
    which basedpyright can't narrow — this Protocol pins the two fields this
    module actually reads so the CUDA tflops estimate stays typed.
    """

    multi_processor_count: int
    max_clock_rate: int


class _CudaModule(Protocol):
    """Static shape for the ``torch.cuda`` submodule surface this file calls.

    ``get_device_properties`` returns a conditionally-assigned dummy type
    when torch is built without CUDA, which basedpyright can't narrow through
    — this Protocol pins the one call this module makes.
    """

    def get_device_properties(self, device: int) -> _CudaDeviceProps: ...


@dataclass
class NodeCapabilityAnnouncement:
    node_id: str
    hostname: str
    platform: str  # darwin / linux
    device_type: str  # apple_silicon / cuda / cpu
    total_ram_gb: float
    available_ram_gb: float
    compute_tflops: float
    mlx_available: bool
    cuda_available: bool
    announced_at: float = field(default_factory=time.time)
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "platform": self.platform,
            "device_type": self.device_type,
            "total_ram_gb": self.total_ram_gb,
            "available_ram_gb": self.available_ram_gb,
            "compute_tflops": self.compute_tflops,
            "mlx_available": self.mlx_available,
            "cuda_available": self.cuda_available,
            "announced_at": self.announced_at,
            "version": self.version,
            "age_s": round(time.time() - self.announced_at, 1),
        }


def _detect_local_capabilities(node_id: str) -> NodeCapabilityAnnouncement:
    """Detect hardware capabilities of the current node."""
    sys_platform = platform.system().lower()
    hostname = platform.node()
    total_ram_gb = 0.0
    compute_tflops = 0.0
    mlx_available = False
    cuda_available = False
    device_type = "cpu"

    # RAM detection
    try:
        import psutil

        total_ram_gb = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        if sys_platform == "darwin":
            try:
                import subprocess

                result = subprocess.run(
                    ["sysctl", "-n", "hw.memsize"],
                    capture_output=True,
                    text=True,
                    timeout=2.0,
                )
                total_ram_gb = int(result.stdout.strip()) / (1024**3)
            except Exception as exc:
                logger.debug("sysctl RAM detection failed: {}", exc)

    # MLX detection
    try:
        import mlx.core as mx

        mlx_available = True
        device_type = "apple_silicon"
        compute_tflops = 10.0  # conservative estimate for Apple Silicon
        del mx
    except ImportError as exc:
        logger.debug("MLX not available: {}", exc)

    # CUDA detection
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_type = "cuda"
            cuda = cast(_CudaModule, cast(Any, torch.cuda))
            props = cuda.get_device_properties(0)
            compute_tflops = (
                props.multi_processor_count * 128 * 2 * props.max_clock_rate * 1e-9
            )
    except ImportError as exc:
        logger.debug("torch not available for CUDA detection: {}", exc)

    return NodeCapabilityAnnouncement(
        node_id=node_id,
        hostname=hostname,
        platform=sys_platform,
        device_type=device_type,
        total_ram_gb=round(total_ram_gb, 2),
        available_ram_gb=round(total_ram_gb * 0.8, 2),  # conservative
        compute_tflops=round(compute_tflops, 2),
        mlx_available=mlx_available,
        cuda_available=cuda_available,
    )


class PeerCapabilityAnnouncer:
    """
    Tracks capability announcements from all nodes in the cluster.
    Master collects these; workers submit via the event system.
    """

    def __init__(self) -> None:
        self._peers: dict[str, NodeCapabilityAnnouncement] = {}
        self._local: NodeCapabilityAnnouncement | None = None

    def announce_local(self, node_id: str) -> NodeCapabilityAnnouncement:
        announcement = _detect_local_capabilities(node_id)
        self._local = announcement
        self._peers[node_id] = announcement
        logger.info(
            f"PeerCapabilityAnnouncer local node={node_id} "
            f"ram={announcement.total_ram_gb:.1f}GB device={announcement.device_type} "
            f"mlx={announcement.mlx_available}"
        )
        return announcement

    def receive_peer(self, announcement: NodeCapabilityAnnouncement) -> None:
        self._peers[announcement.node_id] = announcement
        logger.debug(
            f"PeerCapabilityAnnouncer received peer node={announcement.node_id}"
        )

    def get_peer(self, node_id: str) -> NodeCapabilityAnnouncement | None:
        return self._peers.get(node_id)

    def all_peers(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self._peers.values()]

    def total_cluster_ram_gb(self) -> float:
        return sum(p.total_ram_gb for p in self._peers.values())

    def total_cluster_tflops(self) -> float:
        return sum(p.compute_tflops for p in self._peers.values())

    def cluster_summary(self) -> dict[str, Any]:
        return {
            "peer_count": len(self._peers),
            "total_ram_gb": round(self.total_cluster_ram_gb(), 2),
            "total_compute_tflops": round(self.total_cluster_tflops(), 2),
            "mlx_nodes": sum(1 for p in self._peers.values() if p.mlx_available),
            "cuda_nodes": sum(1 for p in self._peers.values() if p.cuda_available),
        }


PEER_CAPABILITY_ANNOUNCER = PeerCapabilityAnnouncer()
