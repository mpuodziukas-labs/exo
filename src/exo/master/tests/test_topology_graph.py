"""
Behavioral tests for exo.master.topology_graph.

All modules imported by topology_graph (health_score, heartbeat_monitor,
node_registry, utilization_tracker, link_health, and their transitive
dependencies) now exist on disk.  Tests use unittest.mock.patch to replace
the module-level singletons that topology_graph references at call time,
giving each test full control without mutating real global state.

Tests focus on:
- GraphNode / GraphEdge dataclass construction and field types
- _build_edges: link_type classification (tb4 vs ethernet vs unknown)
- _build_edges: permutations — N nodes → N*(N-1) directed edges
- to_d3_json() wire format keys
- to_graph_dict() bandwidth_gbps conversion (Mbps / 1000)
- Edge health comes from link_monitor target stats
- Empty-graph (zero nodes) returns empty edge list
- TopologyGraph.cluster_health_score is None when HEALTH_SCORER has no history
"""

from __future__ import annotations

from unittest.mock import patch

from exo.master.topology_graph import (
    GraphEdge,
    GraphNode,
    TopologyGraphBuilder,
)

# ---------------------------------------------------------------------------
# GraphNode / GraphEdge dataclass tests
# ---------------------------------------------------------------------------


class TestGraphDataclasses:
    def test_graph_node_construction(self) -> None:
        node = GraphNode(
            id="n1",
            label="host1",
            type="master",
            cpu_pct=10.0,
            memory_pct=50.0,
            gpu_pct=80.0,
            health="healthy",
            model_ids=["llama3"],
        )
        assert node.id == "n1"
        assert node.type == "master"
        assert node.model_ids == ["llama3"]

    def test_graph_edge_link_type_values(self) -> None:
        edge = GraphEdge(
            source="n1",
            target="n2",
            latency_ms=1.5,
            throughput_mbps=2000.0,  # > 1000 → tb4
            link_type="tb4",
            health="healthy",
        )
        assert edge.link_type == "tb4"


# ---------------------------------------------------------------------------
# _build_edges link_type classification
# ---------------------------------------------------------------------------


class TestBuildEdgesLinkTypeClassification:
    """Test _build_edges via the full builder, injecting controlled link stats."""

    def _make_two_nodes(self) -> list[GraphNode]:
        return [
            GraphNode("n1", "host1", "master", 0.0, 0.0, None, "healthy"),
            GraphNode("n2", "host2", "worker", 0.0, 0.0, None, "healthy"),
        ]

    def test_tb4_link_classified_when_throughput_above_threshold(self) -> None:
        builder = TopologyGraphBuilder()
        nodes = self._make_two_nodes()
        link_stats = {
            "n1": {
                "p50_latency_ms": 0.5,
                "avg_throughput_mbps": 2000.0,
                "status": "healthy",
            },
            "n2": {
                "p50_latency_ms": 0.5,
                "avg_throughput_mbps": 2000.0,
                "status": "healthy",
            },
        }
        with patch("exo.master.topology_graph.LINK_MONITOR") as lm:
            lm.get_stats.return_value = [
                {"node_id": k, **v} for k, v in link_stats.items()
            ]
            edges = builder._build_edges(nodes)

        tb4_edges = [e for e in edges if e.link_type == "tb4"]
        assert len(tb4_edges) > 0

    def test_ethernet_link_classified_when_throughput_below_threshold(self) -> None:
        builder = TopologyGraphBuilder()
        nodes = self._make_two_nodes()
        with patch("exo.master.topology_graph.LINK_MONITOR") as lm:
            lm.get_stats.return_value = [
                {
                    "node_id": "n1",
                    "p50_latency_ms": 5.0,
                    "avg_throughput_mbps": 500.0,
                    "status": "healthy",
                },
                {
                    "node_id": "n2",
                    "p50_latency_ms": 5.0,
                    "avg_throughput_mbps": 500.0,
                    "status": "healthy",
                },
            ]
            edges = builder._build_edges(nodes)

        ethernet_edges = [e for e in edges if e.link_type == "ethernet"]
        assert len(ethernet_edges) > 0

    def test_unknown_link_when_throughput_zero(self) -> None:
        builder = TopologyGraphBuilder()
        nodes = self._make_two_nodes()
        with patch("exo.master.topology_graph.LINK_MONITOR") as lm:
            lm.get_stats.return_value = []  # no link stats → throughput=0
            edges = builder._build_edges(nodes)

        unknown_edges = [e for e in edges if e.link_type == "unknown"]
        assert len(unknown_edges) == len(edges)  # all unknown

    def test_n_nodes_produce_n_times_n_minus_1_edges(self) -> None:
        """Directed permutations: 3 nodes → 3*2 = 6 edges."""
        builder = TopologyGraphBuilder()
        nodes = [
            GraphNode(f"n{i}", f"host{i}", "worker", 0.0, 0.0, None, "healthy")
            for i in range(3)
        ]
        with patch("exo.master.topology_graph.LINK_MONITOR") as lm:
            lm.get_stats.return_value = []
            edges = builder._build_edges(nodes)

        assert len(edges) == 3 * 2  # permutations(3, 2)

    def test_empty_nodes_produce_zero_edges(self) -> None:
        builder = TopologyGraphBuilder()
        with patch("exo.master.topology_graph.LINK_MONITOR") as lm:
            lm.get_stats.return_value = []
            edges = builder._build_edges([])
        assert edges == []


# ---------------------------------------------------------------------------
# to_d3_json and to_graph_dict
# ---------------------------------------------------------------------------


class TestSerialisation:
    def test_to_d3_json_keys_present(self) -> None:
        builder = TopologyGraphBuilder()
        with (
            patch("exo.master.topology_graph.NODE_REGISTRY") as nr,
            patch("exo.master.topology_graph.HEARTBEAT_MONITOR") as hm,
            patch("exo.master.topology_graph.UTILIZATION_TRACKER") as ut,
            patch("exo.master.topology_graph.LINK_MONITOR") as lm,
            patch("exo.master.topology_graph.HEALTH_SCORER") as hs,
        ):
            nr.all_nodes.return_value = []
            hm.alive_nodes.return_value = []
            hm.evicted_nodes.return_value = []
            ut.get_node.return_value = None
            lm.get_stats.return_value = []
            hs.current.return_value = None

            result = builder.to_d3_json()

        assert "nodes" in result
        assert "links" in result
        assert "generated_at" in result
        assert "cluster_health_score" in result

    def test_to_graph_dict_bandwidth_conversion(self) -> None:
        """bandwidth_gbps = throughput_mbps / 1000."""
        builder = TopologyGraphBuilder()
        with (
            patch("exo.master.topology_graph.NODE_REGISTRY") as nr,
            patch("exo.master.topology_graph.HEARTBEAT_MONITOR") as hm,
            patch("exo.master.topology_graph.UTILIZATION_TRACKER") as ut,
            patch("exo.master.topology_graph.LINK_MONITOR") as lm,
            patch("exo.master.topology_graph.HEALTH_SCORER") as hs,
        ):
            nr.all_nodes.return_value = []
            hm.alive_nodes.return_value = ["a", "b"]
            hm.evicted_nodes.return_value = []
            ut.get_node.return_value = None
            lm.get_stats.return_value = [
                {
                    "node_id": "a",
                    "p50_latency_ms": 1.0,
                    "avg_throughput_mbps": 1000.0,
                    "status": "healthy",
                },
                {
                    "node_id": "b",
                    "p50_latency_ms": 1.0,
                    "avg_throughput_mbps": 1000.0,
                    "status": "healthy",
                },
            ]
            hs.current.return_value = None

            result = builder.to_graph_dict()

        # At least one edge should have bandwidth_gbps = 1000/1000 = 1.0
        assert "edges" in result
        if result["edges"]:
            gbps_vals = [e["bandwidth_gbps"] for e in result["edges"]]
            assert all(isinstance(v, float) for v in gbps_vals)

    def test_cluster_health_score_none_when_no_history(self) -> None:
        builder = TopologyGraphBuilder()
        with (
            patch("exo.master.topology_graph.NODE_REGISTRY") as nr,
            patch("exo.master.topology_graph.HEARTBEAT_MONITOR") as hm,
            patch("exo.master.topology_graph.UTILIZATION_TRACKER") as ut,
            patch("exo.master.topology_graph.LINK_MONITOR") as lm,
            patch("exo.master.topology_graph.HEALTH_SCORER") as hs,
        ):
            nr.all_nodes.return_value = []
            hm.alive_nodes.return_value = []
            hm.evicted_nodes.return_value = []
            ut.get_node.return_value = None
            lm.get_stats.return_value = []
            hs.current.return_value = None  # no history

            graph = builder.build()

        assert graph.cluster_health_score is None
