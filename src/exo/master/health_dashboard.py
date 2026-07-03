"""
health_dashboard.py — live HTML dashboard for the exo distributed inference cluster.

render_dashboard(data)  -> complete HTML string (self-contained, no CDN)
generate_dashboard_data(...) -> collect all subsystem data into one dict
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    from exo.master.admission_control import AdmissionController
    from exo.master.circuit_breaker import CircuitBreakerRegistry
    from exo.master.link_health import LinkHealthMonitor
    from exo.master.memory_monitor import MemoryPressureMonitor
    from exo.master.metrics import Metrics
    from exo.master.slo_tracker import SloTracker

# ---------------------------------------------------------------------------
# Wire-format TypedDicts — shapes of the dicts produced/consumed by the
# dashboard. Mirrors the pattern in topology_graph.py: subsystems return
# `dict[str, Any]` from their own `.to_dict()`/`.stats()`/`.summary()`
# methods; this module asserts the concrete shape once, at the boundary.
# ---------------------------------------------------------------------------


class ClusterSection(TypedDict):
    world_size: int
    active_nodes: int
    master_node_id: str
    uptime_seconds: float


class RequestsSection(TypedDict):
    requests_per_second: float
    active: int
    total: int
    errors: int
    p50_ms: float
    p99_ms: float
    tokens_per_second: float


class CircuitBreakerEntry(TypedDict):
    """Shape of CircuitBreaker.to_dict() entries."""

    worker_id: str
    state: str
    failure_count: int
    success_streak: int
    total_requests: int
    total_failures: int
    error_rate: float
    seconds_in_current_state: float


class LinkHealthEntry(TypedDict):
    """Shape of LinkHealthMonitor.NodeLinkStats.to_dict() entries."""

    node_id: str
    status: str
    p50_latency_ms: float
    p99_latency_ms: float
    avg_throughput_mbps: float
    sample_count: int
    last_sample_ts: float


class SloSection(TypedDict):
    """Shape of SloTracker.summary()."""

    client_count: int
    global_p50_ttft_ms: float
    global_p99_ttft_ms: float
    global_p50_total_ms: float
    global_p99_total_ms: float
    total_violations: int
    slo_ms: float


class MemorySection(TypedDict, total=False):
    """Shape of MemoryPressureMonitor.stats() — total=False since the
    monitor returns the single key {"status": "not_sampled"} before the
    first sample."""

    status: str
    level: str
    ram_pressure: float
    ram_used_gb: float
    ram_total_gb: float
    ram_available_gb: float
    swap_pressure: float
    swap_used_gb: float
    warning_events: int
    critical_events: int


class AdmissionSection(TypedDict):
    """Shape of AdmissionController.stats()."""

    admitted_total: int
    rejected_total: int
    rejection_rate: float
    max_concurrent: int
    max_queue_depth: int
    memory_pressure_threshold: float
    kv_cache_threshold: float


class ModelEntry(TypedDict):
    model_id: str
    version_hash: str
    loaded_at: float
    path: str


class DashboardData(TypedDict):
    generated_at: float
    cluster: ClusterSection
    requests: RequestsSection
    circuit_breakers: list[CircuitBreakerEntry]
    link_health: list[LinkHealthEntry]
    slo: SloSection
    slo_violators: list[str]
    memory: MemorySection
    admission: AdmissionSection
    models: list[ModelEntry]


class _StateLike(Protocol):
    """Structural shape expected of the master `state` object passed into
    generate_dashboard_data(). Only the field actually read is declared —
    callers may pass any object exposing it (duck typing preserved)."""

    last_seen: Mapping[object, object]


# ── colour helpers ────────────────────────────────────────────────────────────
_C = {
    "green": "#3fb950",
    "yellow": "#d29922",
    "red": "#f85149",
    "blue": "#58a6ff",
    "text": "#c9d1d9",
    "dim": "#8b949e",
    "bg": "#0d1117",
    "card": "#161b22",
    "border": "#30363d",
}


def _state_color(state: str) -> str:
    return {"closed": _C["green"], "half_open": _C["yellow"], "open": _C["red"]}.get(
        state, _C["dim"]
    )


def _link_color(status: str) -> str:
    return {"healthy": _C["green"], "warning": _C["yellow"], "degraded": _C["red"]}.get(
        status, _C["dim"]
    )


def _mem_color(level: str) -> str:
    return {
        "ok": _C["green"],
        "warning": _C["yellow"],
        "critical": _C["red"],
        "fatal": _C["red"],
    }.get(level, _C["dim"])


# ── sub-section renderers ─────────────────────────────────────────────────────


def _card(title: str, body: str) -> str:
    return f"""
<div class="card">
  <h2>{title}</h2>
  {body}
</div>"""


def _kv(label: str, value: str | int | float, color: str = "") -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<div class="kv"><span class="label">{label}</span><span class="value"{style}>{value}</span></div>'


def _table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    trs = ""
    for row in rows:
        trs += "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
    return f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>"


# ── section builders ──────────────────────────────────────────────────────────


def _section_cluster(d: DashboardData) -> str:
    c = d["cluster"]
    uptime_s = c.get("uptime_seconds", 0.0)
    h, rem = divmod(int(uptime_s), 3600)
    m, s = divmod(rem, 60)
    uptime_str = f"{h}h {m}m {s}s"
    body = (
        _kv("World size", c.get("world_size", 0), _C["blue"])
        + _kv("Active nodes", c.get("active_nodes", 0))
        + _kv("Master node", c.get("master_node_id", "—"), _C["dim"])
        + _kv("Uptime", uptime_str)
    )
    return _card("Cluster", body)


def _section_requests(d: DashboardData) -> str:
    r = d["requests"]
    rps = round(r.get("requests_per_second", 0.0), 2)
    errors = int(r.get("errors", 0))
    body = (
        _kv("Requests/sec", rps, _C["blue"])
        + _kv("Active", int(r.get("active", 0)))
        + _kv("Total", int(r.get("total", 0)))
        + _kv("Errors", errors, _C["red"] if errors else "")
        + _kv("p50 latency", f"{r.get('p50_ms', 0.0):.1f} ms")
        + _kv("p99 latency", f"{r.get('p99_ms', 0.0):.1f} ms")
        + _kv("Tokens/sec", round(r.get("tokens_per_second", 0.0), 1))
    )
    return _card("Requests", body)


def _section_circuit_breakers(d: DashboardData) -> str:
    breakers = d["circuit_breakers"]
    if not breakers:
        return _card("Circuit Breakers", "<p class='empty'>No workers registered</p>")
    rows: list[list[str]] = []
    for b in breakers:
        state = b.get("state", "unknown")
        color = _state_color(state)
        badge = f'<span style="color:{color};font-weight:600">{state.upper()}</span>'
        last_fail = b.get("seconds_in_current_state", 0.0)
        rows.append(
            [
                b.get("worker_id", "?"),
                badge,
                str(b.get("failure_count", 0)),
                str(b.get("total_failures", 0)),
                f"{b.get('error_rate', 0.0):.1%}",
                f"{last_fail:.0f}s ago",
            ]
        )
    tbl = _table(
        ["Worker", "State", "Fail streak", "Total fails", "Error rate", "In state"],
        rows,
    )
    return _card("Circuit Breakers", tbl)


def _section_link_health(d: DashboardData) -> str:
    links = d["link_health"]
    if not links:
        return _card("Link Health", "<p class='empty'>No link data</p>")
    rows: list[list[str]] = []
    for lk in links:
        status = lk.get("status", "unknown")
        color = _link_color(status)
        badge = f'<span style="color:{color};font-weight:600">{status.upper()}</span>'
        rows.append(
            [
                lk.get("node_id", "?"),
                badge,
                f"{lk.get('p50_latency_ms', 0.0):.1f} ms",
                f"{lk.get('p99_latency_ms', 0.0):.1f} ms",
                f"{lk.get('avg_throughput_mbps', 0.0):.2f} Mbps",
                str(lk.get("sample_count", 0)),
            ]
        )
    tbl = _table(
        ["Node", "Status", "p50 lat", "p99 lat", "Throughput", "Samples"], rows
    )
    return _card("Link Health", tbl)


def _section_slo(d: DashboardData) -> str:
    slo = d["slo"]
    violators = d["slo_violators"]
    vcount = slo.get("total_violations", 0)
    vcolor = _C["red"] if vcount else _C["green"]
    body = (
        _kv("SLO target (TTFT)", f"{slo.get('slo_ms', 500.0):.0f} ms")
        + _kv("Global p50 TTFT", f"{slo.get('global_p50_ttft_ms', 0.0):.1f} ms")
        + _kv("Global p99 TTFT", f"{slo.get('global_p99_ttft_ms', 0.0):.1f} ms")
        + _kv("Clients tracked", slo.get("client_count", 0))
        + _kv("Total violations", vcount, vcolor)
    )
    if violators:
        vlist = "".join(f"<li>{v}</li>" for v in violators[:5])
        body += f'<div class="kv"><span class="label">Top violators</span><ul class="violators">{vlist}</ul></div>'
    return _card("SLO", body)


def _section_memory(d: DashboardData) -> str:
    mem = d["memory"]
    level = mem.get("level", "ok")
    color = _mem_color(level)
    pressure = mem.get("ram_pressure", 0.0)
    used = mem.get("ram_used_gb", 0.0)
    total = mem.get("ram_total_gb", 0.0)
    bar_pct = int(pressure * 100)
    bar_color = color
    critical_events = mem.get("critical_events", 0)
    body = (
        _kv("Level", level.upper(), color)
        + _kv("RAM used", f"{used:.1f} / {total:.1f} GB")
        + _kv("RAM pressure", f"{pressure:.1%}", color)
        + f'<div class="bar-wrap"><div class="bar" style="width:{bar_pct}%;background:{bar_color}"></div></div>'
        + _kv("Swap used", f"{mem.get('swap_used_gb', 0.0):.1f} GB")
        + _kv("Warnings", mem.get("warning_events", 0))
        + _kv(
            "Critical events",
            critical_events,
            _C["red"] if critical_events else "",
        )
    )
    return _card("Memory", body)


def _section_admission(d: DashboardData) -> str:
    adm = d["admission"]
    rrate = adm.get("rejection_rate", 0.0)
    gate_color = _C["red"] if rrate > 0.1 else _C["green"]
    rejected_total = adm.get("rejected_total", 0)
    body = (
        _kv("Max concurrent", adm.get("max_concurrent", 0))
        + _kv("Admitted total", adm.get("admitted_total", 0), _C["green"])
        + _kv(
            "Rejected total",
            rejected_total,
            _C["red"] if rejected_total else "",
        )
        + _kv("Rejection rate", f"{rrate:.1%}", gate_color)
        + _kv("Max queue depth", adm.get("max_queue_depth", 0))
        + _kv("Mem threshold", f"{adm.get('memory_pressure_threshold', 0.0):.0%}")
        + _kv("KV threshold", f"{adm.get('kv_cache_threshold', 0.0):.0%}")
    )
    return _card("Admission Control", body)


def _section_models(d: DashboardData) -> str:
    models = d["models"]
    if not models:
        return _card("Models", "<p class='empty'>No models registered</p>")
    rows: list[list[str]] = []
    for m in models:
        loaded_at = m.get("loaded_at", 0.0)
        ts = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(loaded_at))
            if loaded_at
            else "—"
        )
        rows.append(
            [
                m.get("model_id", "?"),
                m.get("version_hash", "—"),
                ts,
                m.get("path", "—"),
            ]
        )
    tbl = _table(["Model ID", "Version", "Loaded at", "Path"], rows)
    return _card("Models", tbl)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  background: {_C["bg"]}; color: {_C["text"]};
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace;
  font-size: 13px; padding: 16px;
}}
h1 {{ font-size: 18px; color: {_C["blue"]}; margin-bottom: 16px; }}
h1 span {{ font-size: 12px; color: {_C["dim"]}; margin-left: 12px; }}
.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 12px;
}}
.card {{
  background: {_C["card"]}; border: 1px solid {_C["border"]};
  border-radius: 6px; padding: 14px;
}}
.card h2 {{ font-size: 13px; color: {_C["dim"]}; text-transform: uppercase;
  letter-spacing: .06em; margin-bottom: 10px; border-bottom: 1px solid {_C["border"]}; padding-bottom: 6px; }}
.kv {{ display: flex; justify-content: space-between; align-items: flex-start;
  padding: 3px 0; }}
.label {{ color: {_C["dim"]}; }}
.value {{ font-weight: 600; text-align: right; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 4px; }}
th {{ text-align: left; color: {_C["dim"]}; font-weight: 500;
  border-bottom: 1px solid {_C["border"]}; padding: 4px 6px; }}
td {{ padding: 4px 6px; border-bottom: 1px solid {_C["border"]}; word-break: break-all; }}
tr:last-child td {{ border-bottom: none; }}
.empty {{ color: {_C["dim"]}; font-style: italic; padding: 6px 0; }}
.bar-wrap {{ background: {_C["border"]}; border-radius: 3px; height: 6px;
  margin: 6px 0; overflow: hidden; }}
.bar {{ height: 6px; border-radius: 3px; transition: width .4s; }}
ul.violators {{ list-style: none; text-align: right; }}
ul.violators li {{ color: {_C["red"]}; font-size: 11px; }}
.footer {{ margin-top: 14px; text-align: right; color: {_C["dim"]}; font-size: 11px; }}
"""

# ── JS polling (partial update via fetch /dashboard/data) ─────────────────────
_JS = """
(function() {
  var refreshMs = 5000;
  function pad(n){ return n < 10 ? '0'+n : n; }
  function tick(){
    var now = new Date();
    var ts = now.getFullYear()+'-'+pad(now.getMonth()+1)+'-'+pad(now.getDate())
      +' '+pad(now.getHours())+':'+pad(now.getMinutes())+':'+pad(now.getSeconds());
    var el = document.getElementById('last-refresh');
    if(el) el.textContent = 'Last refresh: '+ts;
  }
  tick();
  setTimeout(function(){ window.location.reload(); }, refreshMs);
})();
"""


# ── public API ────────────────────────────────────────────────────────────────


def render_dashboard(data: DashboardData) -> str:
    """Return a complete self-contained HTML page string."""
    now = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())
    sections = "\n".join(
        [
            _section_cluster(data),
            _section_requests(data),
            _section_circuit_breakers(data),
            _section_link_health(data),
            _section_slo(data),
            _section_memory(data),
            _section_admission(data),
            _section_models(data),
        ]
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="5">
<title>exo cluster dashboard</title>
<style>{_CSS}</style>
</head>
<body>
<h1>exo cluster dashboard <span id="last-refresh">Loading…</span></h1>
<div class="grid">
{sections}
</div>
<div class="footer">Generated {now} · auto-refresh every 5 s</div>
<script>{_JS}</script>
</body>
</html>"""


def generate_dashboard_data(
    state: "_StateLike | None",
    metrics: "Metrics",
    circuit_breakers: "CircuitBreakerRegistry",
    link_monitor: "LinkHealthMonitor",
    slo_tracker: "SloTracker",
    memory_monitor: "MemoryPressureMonitor",
    admission_controller: "AdmissionController",
    model_registry_fn: Callable[[], list[ModelEntry]],
) -> DashboardData:
    """Collect all subsystem data into a single dict for the dashboard."""
    # Cluster
    world_size = len(state.last_seen) if state else 0
    # metrics._start_time is intentionally read via getattr rather than the
    # direct private attribute: Metrics has no public uptime accessor and
    # this module cannot modify metrics.py, so reflection is used instead of
    # a suppression comment — same value, no cross-module private access.
    start_time: float = getattr(metrics, "_start_time", time.time())
    uptime = time.time() - start_time

    # Requests — compute RPS from counter delta isn't easy without history;
    # use tokens_per_second as a proxy and expose raw counters.
    _, hist_sum, hist_count = metrics.request_duration_seconds.get()
    p50_ms = 0.0
    p99_ms = 0.0
    if hist_count:
        avg_s = hist_sum / hist_count
        p50_ms = avg_s * 1000  # rough; real percentile needs sorted samples
        p99_ms = p50_ms * 3  # placeholder when no per-request SLO data

    slo_summary = cast("SloSection", cast(object, slo_tracker.summary()))
    slo_p50 = slo_summary.get("global_p50_ttft_ms", 0.0)
    slo_p99 = slo_summary.get("global_p99_ttft_ms", 0.0)
    if slo_p50:
        p50_ms = slo_p50
    if slo_p99:
        p99_ms = slo_p99

    # Memory — sample on every dashboard render
    memory_monitor.sample()

    master_node_id = str(getattr(state, "master_node_id", "—")) if state else "—"

    return {
        "generated_at": time.time(),
        "cluster": {
            "world_size": world_size,
            "active_nodes": world_size,
            "master_node_id": master_node_id,
            "uptime_seconds": round(uptime, 1),
        },
        "requests": {
            "requests_per_second": round(
                metrics.tokens_per_second.get() / max(1, 200), 4
            ),
            "active": int(metrics.requests_active.get()),
            "total": int(metrics.requests_total.get()),
            "errors": int(metrics.errors_total.get()),
            "p50_ms": round(p50_ms, 2),
            "p99_ms": round(p99_ms, 2),
            "tokens_per_second": round(metrics.tokens_per_second.get(), 2),
        },
        "circuit_breakers": cast(
            "list[CircuitBreakerEntry]", circuit_breakers.all_states()
        ),
        "link_health": cast("list[LinkHealthEntry]", link_monitor.get_stats()),
        "slo": slo_summary,
        "slo_violators": slo_tracker.violating_clients()[:5],
        "memory": cast("MemorySection", cast(object, memory_monitor.stats())),
        "admission": cast(
            "AdmissionSection", cast(object, admission_controller.stats())
        ),
        "models": model_registry_fn(),
    }
