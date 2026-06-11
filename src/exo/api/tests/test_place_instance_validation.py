"""Placement feasibility must be validated synchronously at the API boundary.

Without this, POST /place_instance accepts commands that can never be
satisfied by the current topology — the failure happens asynchronously in the
master's command processor and the client only ever sees "Command received."
"""

import pytest
from fastapi import HTTPException

from exo.api.main import ensure_placement_feasible
from exo.shared.models.model_cards import ModelCard, ModelId, ModelTask
from exo.shared.topology import Topology
from exo.shared.types.commands import PlaceInstance
from exo.shared.types.common import CommandId, NodeId
from exo.shared.types.memory import Memory
from exo.shared.types.profiling import (
    MemoryUsage,
    NetworkInterfaceInfo,
    NodeNetworkInfo,
)
from exo.shared.types.state import State
from exo.shared.types.worker.instances import InstanceMeta
from exo.shared.types.worker.shards import Sharding


def _state_with_single_node(available_kb: int) -> State:
    topology = Topology()
    node_id = NodeId()
    topology.add_node(node_id)
    return State(
        topology=topology,
        node_memory={
            node_id: MemoryUsage.from_bytes(
                ram_total=available_kb * 1024,
                ram_available=available_kb * 1024,
                swap_total=0,
                swap_available=0,
            )
        },
        node_network={
            node_id: NodeNetworkInfo(
                interfaces=[NetworkInterfaceInfo(name="en0", ip_address="169.254.0.1")]
            )
        },
    )


def _command(storage_kb: int, min_nodes: int = 1) -> PlaceInstance:
    return PlaceInstance(
        command_id=CommandId(),
        model_card=ModelCard(
            model_id=ModelId("test-model"),
            storage_size=Memory.from_kb(storage_kb),
            n_layers=10,
            hidden_size=32,
            supports_tensor=True,
            tasks=[ModelTask.TextGeneration],
        ),
        sharding=Sharding.Pipeline,
        instance_meta=InstanceMeta.MlxRing,
        min_nodes=min_nodes,
    )


def test_infeasible_placement_raises_http_400() -> None:
    state = _state_with_single_node(available_kb=1000)
    with pytest.raises(HTTPException) as excinfo:
        ensure_placement_feasible(_command(storage_kb=100_000), state)
    assert excinfo.value.status_code == 400
    assert "No cycles found" in str(excinfo.value.detail)


def test_min_nodes_unsatisfiable_raises_http_400() -> None:
    # One connected node can never satisfy min_nodes=2.
    state = _state_with_single_node(available_kb=10_000_000)
    with pytest.raises(HTTPException) as excinfo:
        ensure_placement_feasible(_command(storage_kb=100, min_nodes=2), state)
    assert excinfo.value.status_code == 400


def test_feasible_placement_does_not_raise() -> None:
    state = _state_with_single_node(available_kb=10_000_000)
    ensure_placement_feasible(_command(storage_kb=100), state)
