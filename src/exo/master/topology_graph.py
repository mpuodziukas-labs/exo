"""topology_graph.py — D3.js-ready cluster topology graph data for exo.

Generates graph nodes (cluster members), latency/throughput edges, and model-
shard distribution.  Also renders a self-contained dark-themed HTML visualization.

Public API
----------
GraphNode         — per-node display record
GraphEdge         — per-link display record
TopologyGraph     — full snapshot
TopologyGraphBuilder.build()       → TopologyGraph
TopologyGraphBuilder.to_d3_json()  → dict   (D3 force-graph wire format)
TopologyGraphBuilder.render_html() → str    (standalone HTML page)
TopologyGraphBuilder.stats()       → dict
TOPOLOGY_BUILDER   — module-level singleton
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Final, Literal, TypedDict

from loguru import logger

from exo.master.health_score import HEALTH_SCORER
from exo.master.heartbeat_monitor import HEARTBEAT_MONITOR
from exo.master.link_health import LINK_MONITOR
from exo.master.node_registry import NODE_REGISTRY
from exo.master.utilization_tracker import UTILIZATION_TRACKER

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

NodeType = Literal["master", "worker", "unknown"]
LinkType = Literal["tb4", "ethernet", "unknown"]

_TB4_THRESHOLD_MBPS: Final[float] = 1_000.0  # >1 Gbps → classify as TB4


@dataclass
class GraphNode:
    id: str
    label: str
    type: NodeType
    cpu_pct: float
    memory_pct: float
    gpu_pct: float | None
    health: str
    model_ids: list[str] = field(default_factory=list)


@dataclass
class GraphEdge:
    source: str
    target: str
    latency_ms: float
    throughput_mbps: float
    link_type: LinkType
    health: str


@dataclass
class TopologyGraph:
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    generated_at: float
    cluster_health_score: float | None


# ---------------------------------------------------------------------------
# Wire-format TypedDicts (return shapes for the JSON/D3/visualization APIs)
# ---------------------------------------------------------------------------


class D3Node(TypedDict):
    id: str
    label: str
    type: NodeType
    cpu_pct: float
    memory_pct: float
    gpu_pct: float | None
    health: str
    model_ids: list[str]


class D3Link(TypedDict):
    source: str
    target: str
    latency_ms: float
    throughput_mbps: float
    link_type: LinkType
    health: str


class D3Graph(TypedDict):
    nodes: list[D3Node]
    links: list[D3Link]
    generated_at: float
    cluster_health_score: float | None


class VisNode(TypedDict):
    id: str
    label: str
    type: NodeType
    ram_gb: float
    status: str


class VisEdge(TypedDict):
    source: str
    target: str
    bandwidth_gbps: float
    type: str


class GraphMetadata(TypedDict):
    node_count: int
    edge_count: int


class VisGraphDict(TypedDict):
    nodes: list[VisNode]
    edges: list[VisEdge]
    metadata: GraphMetadata


class TopologyStats(TypedDict):
    node_count: int
    edge_count: int
    node_health_counts: dict[str, int]
    link_type_counts: dict[str, int]
    cluster_health_score: float | None
    generated_at: float


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class TopologyGraphBuilder:
    """Assembles a TopologyGraph from live singleton data sources."""

    # ------------------------------------------------------------------
    # Core build
    # ------------------------------------------------------------------

    def build(self) -> TopologyGraph:
        """Snapshot current cluster state into a TopologyGraph."""
        nodes = self._build_nodes()
        edges = self._build_edges(nodes)
        health_report = HEALTH_SCORER.current()
        health_score: float | None = (
            health_report.overall_score if health_report is not None else None
        )
        graph = TopologyGraph(
            nodes=nodes,
            edges=edges,
            generated_at=time.time(),
            cluster_health_score=health_score,
        )
        logger.debug(
            f"[topology_graph] built snapshot: "
            f"nodes={len(nodes)} edges={len(edges)} "
            f"health={health_score}"
        )
        return graph

    # ------------------------------------------------------------------
    # Helpers — nodes
    # ------------------------------------------------------------------

    def _build_nodes(self) -> list[GraphNode]:
        registry_caps = {cap.node_id: cap for cap in NODE_REGISTRY.all_nodes()}
        alive_ids: set[str] = set(HEARTBEAT_MONITOR.alive_nodes())

        # Union of known node IDs
        all_ids: set[str] = set(registry_caps.keys()) | alive_ids

        if not all_ids:
            logger.warning(
                "[topology_graph] no nodes found in registry or heartbeat monitor"
            )
            return []

        # Master = node with highest flops among registry entries;
        # fall back to the lexicographically first id when registry is empty.
        master_id: str | None = None
        if registry_caps:
            master_id = max(
                registry_caps,
                key=lambda nid: registry_caps[nid].compute_flops_tflops,
            )

        nodes: list[GraphNode] = []
        for node_id in sorted(all_ids):
            cap = registry_caps.get(node_id)
            node_util = UTILIZATION_TRACKER.get_node(node_id)

            # Resolve utilization
            cpu_pct = 0.0
            memory_pct = 0.0
            gpu_pct: float | None = None
            if node_util is not None:
                util_dict = node_util.to_dict()
                latest = util_dict.get("latest")
                if latest is not None:
                    cpu_pct = latest["cpu_pct"]
                    memory_pct = latest["memory_pct"]
                    gpu_pct = latest["gpu_pct"]  # None when unavailable

            # Node type
            if master_id is not None and node_id == master_id:
                node_type: NodeType = "master"
            elif node_id in alive_ids or node_id in registry_caps:
                node_type = "worker"
            else:
                node_type = "unknown"

            # Health: evicted = dead, alive + in registry = healthy, else unknown
            evicted_ids: set[str] = set(HEARTBEAT_MONITOR.evicted_nodes())
            if node_id in evicted_ids:
                health = "dead"
            elif node_id in alive_ids:
                health = "healthy"
            else:
                health = "unknown"

            label = cap.hostname if cap else node_id[:12]
            model_ids = list(cap.loaded_models) if cap else []

            nodes.append(
                GraphNode(
                    id=node_id,
                    label=label,
                    type=node_type,
                    cpu_pct=round(cpu_pct, 1),
                    memory_pct=round(memory_pct, 1),
                    gpu_pct=round(gpu_pct, 1) if gpu_pct is not None else None,
                    health=health,
                    model_ids=model_ids,
                )
            )

        return nodes

    # ------------------------------------------------------------------
    # Helpers — edges
    # ------------------------------------------------------------------

    def _build_edges(self, nodes: list[GraphNode]) -> list[GraphEdge]:
        """One directed edge per ordered node pair using LINK_MONITOR stats."""
        link_stats: dict[str, dict[str, Any]] = {
            s["node_id"] if isinstance(s["node_id"], str) else "": s
            for s in LINK_MONITOR.get_stats()
        }
        node_ids = [n.id for n in nodes]
        edges: list[GraphEdge] = []

        for source_id, target_id in itertools.permutations(node_ids, 2):
            # Use target's link stats (the monitor tracks per-target latency)
            stats = link_stats.get(target_id)
            latency_ms: float = (
                stats["p50_latency_ms"]
                if stats and isinstance(stats["p50_latency_ms"], float)
                else 0.0
            )
            throughput_mbps: float = (
                stats["avg_throughput_mbps"]
                if stats and isinstance(stats["avg_throughput_mbps"], float)
                else 0.0
            )
            edge_health: str = (
                stats["status"]
                if stats and isinstance(stats["status"], str)
                else "unknown"
            )

            link_type: LinkType = (
                "tb4"
                if throughput_mbps > _TB4_THRESHOLD_MBPS
                else "ethernet"
                if throughput_mbps > 0
                else "unknown"
            )

            edges.append(
                GraphEdge(
                    source=source_id,
                    target=target_id,
                    latency_ms=round(latency_ms, 2),
                    throughput_mbps=round(throughput_mbps, 2),
                    link_type=link_type,
                    health=edge_health,
                )
            )

        return edges

    # ------------------------------------------------------------------
    # D3 JSON serialisation
    # ------------------------------------------------------------------

    def to_d3_json(self) -> D3Graph:
        """Return D3 force-directed graph format: {nodes: [...], links: [...]}."""
        graph = self.build()

        d3_nodes: list[D3Node] = [
            {
                "id": n.id,
                "label": n.label,
                "type": n.type,
                "cpu_pct": n.cpu_pct,
                "memory_pct": n.memory_pct,
                "gpu_pct": n.gpu_pct,
                "health": n.health,
                "model_ids": n.model_ids,
            }
            for n in graph.nodes
        ]

        d3_links: list[D3Link] = [
            {
                "source": e.source,
                "target": e.target,
                "latency_ms": e.latency_ms,
                "throughput_mbps": e.throughput_mbps,
                "link_type": e.link_type,
                "health": e.health,
            }
            for e in graph.edges
        ]

        return {
            "nodes": d3_nodes,
            "links": d3_links,
            "generated_at": graph.generated_at,
            "cluster_health_score": graph.cluster_health_score,
        }

    # ------------------------------------------------------------------
    # HTML visualisation
    # ------------------------------------------------------------------

    def render_html(self) -> str:
        """Render a self-contained dark-theme D3 force-directed graph HTML page."""
        import json as _json

        data = self.to_d3_json()
        data_json = _json.dumps(data, indent=2)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>exo Cluster Topology</title>
<script src="https://d3js.org/d3.v7.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: #0d1117;
    color: #c9d1d9;
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 13px;
    overflow: hidden;
  }}
  #header {{
    padding: 10px 16px;
    background: #161b22;
    border-bottom: 1px solid #30363d;
    display: flex;
    align-items: center;
    gap: 16px;
  }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: #58a6ff; }}
  #health-badge {{
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
  }}
  #svg-container {{ width: 100vw; height: calc(100vh - 44px); }}
  svg {{ width: 100%; height: 100%; }}

  .node circle {{
    stroke-width: 2;
    cursor: pointer;
    transition: r 0.15s ease;
  }}
  .node circle:hover {{ opacity: 0.85; }}
  .node text {{
    fill: #c9d1d9;
    font-size: 11px;
    pointer-events: none;
    dominant-baseline: central;
  }}
  .node .sub-label {{
    fill: #8b949e;
    font-size: 9px;
  }}

  .link {{
    stroke-opacity: 0.6;
    fill: none;
    transition: stroke-opacity 0.2s;
  }}
  .link:hover {{ stroke-opacity: 1.0; }}

  #tooltip {{
    position: absolute;
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 10px 14px;
    pointer-events: none;
    font-size: 11px;
    line-height: 1.6;
    color: #c9d1d9;
    min-width: 160px;
    display: none;
    z-index: 100;
  }}
  #tooltip .tt-title {{ color: #58a6ff; font-weight: 600; margin-bottom: 4px; }}
</style>
</head>
<body>
<div id="header">
  <h1>&#x25CB; exo Cluster Topology</h1>
  <span id="health-badge">Loading...</span>
  <span id="ts" style="margin-left:auto; color:#8b949e; font-size:11px;"></span>
</div>
<div id="svg-container">
  <svg id="graph"></svg>
</div>
<div id="tooltip"></div>

<script>
const RAW = {data_json};

// ── health badge ─────────────────────────────────────────────────────────────
const badge = document.getElementById('health-badge');
const score = RAW.cluster_health_score;
if (score !== null && score !== undefined) {{
  const col = score >= 90 ? '#3fb950' : score >= 75 ? '#d29922' : score >= 50 ? '#f85149' : '#6e7681';
  badge.textContent = `Health: ${{score.toFixed(1)}}`;
  badge.style.background = col + '33';
  badge.style.color = col;
  badge.style.border = `1px solid ${{col}}66`;
}} else {{
  badge.textContent = 'Health: n/a';
  badge.style.color = '#8b949e';
}}
const ts = RAW.generated_at ? new Date(RAW.generated_at * 1000).toLocaleTimeString() : '';
document.getElementById('ts').textContent = ts ? `Snapshot: ${{ts}}` : '';

// ── helpers ──────────────────────────────────────────────────────────────────
const HEALTH_COLOR = {{ healthy: '#3fb950', warning: '#d29922', degraded: '#f85149', dead: '#6e7681', unknown: '#58a6ff' }};
const NODE_FILL    = {{ master: '#388bfd', worker: '#2ea043', unknown: '#6e7681' }};
const LINK_COLOR   = {{ healthy: '#3fb950', warning: '#d29922', degraded: '#f85149', unknown: '#58a6ff' }};

// ── layout ───────────────────────────────────────────────────────────────────
const W = window.innerWidth;
const H = window.innerHeight - 44;
const svg = d3.select('#graph')
  .attr('viewBox', `0 0 ${{W}} ${{H}}`);

// defs: arrowhead marker per health colour
const defs = svg.append('defs');
['healthy','warning','degraded','unknown'].forEach(h => {{
  defs.append('marker')
    .attr('id', `arrow-${{h}}`)
    .attr('viewBox', '0 -5 10 10')
    .attr('refX', 22)
    .attr('refY', 0)
    .attr('markerWidth', 6)
    .attr('markerHeight', 6)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-5L10,0L0,5')
    .attr('fill', LINK_COLOR[h] || '#58a6ff');
}});

const container = svg.append('g');
svg.call(d3.zoom().scaleExtent([0.2, 4]).on('zoom', e => container.attr('transform', e.transform)));

// ── simulation ───────────────────────────────────────────────────────────────
const nodes = RAW.nodes.map(d => ({{ ...d }}));
const links = RAW.links.map(d => ({{ ...d }}));

const sim = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links).id(d => d.id).distance(120).strength(0.4))
  .force('charge', d3.forceManyBody().strength(-400))
  .force('center', d3.forceCenter(W / 2, H / 2))
  .force('collision', d3.forceCollide(40));

// ── links ────────────────────────────────────────────────────────────────────
const link = container.append('g').attr('class', 'links')
  .selectAll('line')
  .data(links)
  .join('line')
  .attr('class', 'link')
  .attr('stroke', d => LINK_COLOR[d.health] || '#444')
  .attr('stroke-width', d => Math.max(1, Math.min(6, d.throughput_mbps / 200)))
  .attr('marker-end', d => `url(#arrow-${{d.health || 'unknown'}})`);

// ── node groups ──────────────────────────────────────────────────────────────
const tooltip = document.getElementById('tooltip');

const node = container.append('g').attr('class', 'nodes')
  .selectAll('g')
  .data(nodes)
  .join('g')
  .attr('class', 'node')
  .call(d3.drag()
    .on('start', (e, d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }})
    .on('drag',  (e, d) => {{ d.fx = e.x; d.fy = e.y; }})
    .on('end',   (e, d) => {{ if (!e.active) sim.alphaTarget(0); d.fx = null; d.fy = null; }}))
  .on('mouseover', (e, d) => {{
    const gpu = d.gpu_pct !== null && d.gpu_pct !== undefined ? `${{d.gpu_pct}}%` : 'n/a';
    const models = d.model_ids && d.model_ids.length ? d.model_ids.join('<br/>&#x2023; ') : '—';
    tooltip.innerHTML = `
      <div class="tt-title">${{d.label}}</div>
      <div>Type: <b>${{d.type}}</b></div>
      <div>Health: <b style="color:${{HEALTH_COLOR[d.health] || '#ccc'}}">${{d.health}}</b></div>
      <div>CPU: ${{d.cpu_pct}}% &nbsp; MEM: ${{d.memory_pct}}% &nbsp; GPU: ${{gpu}}</div>
      <div style="margin-top:4px">Models:<br/>&#x2023; ${{models}}</div>`;
    tooltip.style.display = 'block';
    tooltip.style.left = (e.pageX + 14) + 'px';
    tooltip.style.top  = (e.pageY - 10) + 'px';
  }})
  .on('mousemove', e => {{
    tooltip.style.left = (e.pageX + 14) + 'px';
    tooltip.style.top  = (e.pageY - 10) + 'px';
  }})
  .on('mouseleave', () => {{ tooltip.style.display = 'none'; }});

node.append('circle')
  .attr('r', d => d.type === 'master' ? 22 : 16)
  .attr('fill', d => NODE_FILL[d.type] || '#444')
  .attr('stroke', d => HEALTH_COLOR[d.health] || '#444');

node.append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', '-26px')
  .attr('class', 'label')
  .text(d => d.label.length > 14 ? d.label.slice(0, 13) + '…' : d.label);

node.append('text')
  .attr('text-anchor', 'middle')
  .attr('dy', '-15px')
  .attr('class', 'sub-label')
  .text(d => d.type.toUpperCase());

// ── tick ─────────────────────────────────────────────────────────────────────
sim.on('tick', () => {{
  link
    .attr('x1', d => d.source.x)
    .attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x)
    .attr('y2', d => d.target.y);
  node.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});
</script>
</body>
</html>"""

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def to_graph_dict(self) -> VisGraphDict:
        """Return graph data in the standard visualization format.

        Returns a dict with ``nodes``, ``edges``, and ``metadata`` keys that is
        suitable for d3.js, Cytoscape, or any graph visualization library.

        Node fields: id, label, type, ram_gb, status
        Edge fields: source, target, bandwidth_gbps, type
        """
        registry_caps = {cap.node_id: cap for cap in NODE_REGISTRY.all_nodes()}
        graph = self.build()

        vis_nodes: list[VisNode] = []
        for n in graph.nodes:
            cap = registry_caps.get(n.id)
            ram_gb: float = round(cap.ram_total_gb, 1) if cap else 0.0
            vis_nodes.append(
                {
                    "id": n.id,
                    "label": n.label,
                    "type": n.type,
                    "ram_gb": ram_gb,
                    "status": n.health,
                }
            )

        vis_edges: list[VisEdge] = []
        for e in graph.edges:
            bandwidth_gbps = round(e.throughput_mbps / 1_000.0, 3)
            edge_type: str = (
                "thunderbolt"
                if e.link_type == "tb4"
                else "ethernet"
                if e.link_type == "ethernet"
                else "wifi"
            )
            vis_edges.append(
                {
                    "source": e.source,
                    "target": e.target,
                    "bandwidth_gbps": bandwidth_gbps,
                    "type": edge_type,
                }
            )

        return {
            "nodes": vis_nodes,
            "edges": vis_edges,
            "metadata": {
                "node_count": len(vis_nodes),
                "edge_count": len(vis_edges),
            },
        }

    def stats(self) -> TopologyStats:
        graph = self.build()
        health_counts: dict[str, int] = {}
        for n in graph.nodes:
            health_counts[n.health] = health_counts.get(n.health, 0) + 1

        link_type_counts: dict[str, int] = {}
        for e in graph.edges:
            link_type_counts[e.link_type] = link_type_counts.get(e.link_type, 0) + 1

        return {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "node_health_counts": health_counts,
            "link_type_counts": link_type_counts,
            "cluster_health_score": graph.cluster_health_score,
            "generated_at": graph.generated_at,
        }


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

TOPOLOGY_BUILDER: Final[TopologyGraphBuilder] = TopologyGraphBuilder()
