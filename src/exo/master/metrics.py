# Hand-rolled Prometheus-format metrics — no external dependency
import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class _Counter:
    value: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class _Gauge:
    value: float = 0.0
    _lock: Lock = field(default_factory=Lock)

    def set(self, value: float) -> None:
        with self._lock:
            self.value = value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self.value -= amount

    def get(self) -> float:
        with self._lock:
            return self.value


@dataclass
class _Histogram:
    buckets: tuple[float, ...] = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    )
    _counts: dict[float, int] = field(default_factory=dict)
    _sum: float = 0.0
    _total: int = 0
    _lock: Lock = field(default_factory=Lock)

    def __post_init__(self) -> None:
        self._counts = {b: 0 for b in self.buckets}

    def observe(self, value: float) -> None:
        with self._lock:
            self._sum += value
            self._total += 1
            for b in self.buckets:
                if value <= b:
                    self._counts[b] += 1

    def get(self) -> tuple[dict[float, int], float, int]:
        with self._lock:
            return dict(self._counts), self._sum, self._total


class Metrics:
    def __init__(self) -> None:
        self.requests_total = _Counter()
        self.requests_active = _Gauge()
        self.tokens_generated_total = _Counter()
        self.request_duration_seconds = _Histogram()
        self.tokens_per_second = _Gauge()
        self.world_size = _Gauge()
        self.errors_total = _Counter()
        # SLO / alerting gauges
        self.memory_pressure_ratio = (
            _Gauge()
        )  # 0.0-1.0 RAM pressure (mirrors MemoryPressureMonitor)
        self.circuit_breaker_open = _Gauge()  # 1 if any circuit breaker is OPEN, else 0
        self.priority_queue_depth = _Gauge()  # current number of waiting requests
        self.rate_limit_rejected_total = (
            _Counter()
        )  # cumulative requests rejected by rate limiter
        # Latency percentiles (milliseconds) — updated by inference pipeline
        self.inference_latency_p50_ms = _Gauge()  # rolling p50 latency in ms
        self.inference_latency_p99_ms = _Gauge()  # rolling p99 latency in ms
        # Load shedding counter (mirrors SheddingController for convenience)
        self.load_shed_total = _Counter()  # cumulative requests shed
        # KV cache hit rate (0.0-1.0) — updated by prefix cache / KV tier
        self.kv_cache_hit_rate = _Gauge()  # fraction of KV lookups that hit
        # Memory pressure alias (same semantics as memory_pressure_ratio, without _ratio suffix)
        self.memory_pressure = _Gauge()  # 0.0-1.0 RAM pressure
        # Error budget and SLO
        self.error_budget_remaining_pct = _Gauge()  # % of error budget not yet consumed
        self.slo_violation_rate = _Gauge()  # aggregate SLO violation rate
        self._start_time = time.time()

    def render(self) -> str:
        lines: list[str] = []
        uptime = time.time() - self._start_time

        lines.append("# HELP exo_uptime_seconds Time since exo started")
        lines.append("# TYPE exo_uptime_seconds gauge")
        lines.append(f"exo_uptime_seconds {uptime:.3f}")

        lines.append("# HELP exo_requests_total Total inference requests received")
        lines.append("# TYPE exo_requests_total counter")
        lines.append(f"exo_requests_total {self.requests_total.get():.0f}")

        lines.append("# HELP exo_requests_active Currently active inference requests")
        lines.append("# TYPE exo_requests_active gauge")
        lines.append(f"exo_requests_active {self.requests_active.get():.0f}")

        lines.append("# HELP exo_tokens_generated_total Total tokens generated")
        lines.append("# TYPE exo_tokens_generated_total counter")
        lines.append(
            f"exo_tokens_generated_total {self.tokens_generated_total.get():.0f}"
        )

        lines.append("# HELP exo_tokens_per_second Current token generation rate")
        lines.append("# TYPE exo_tokens_per_second gauge")
        lines.append(f"exo_tokens_per_second {self.tokens_per_second.get():.3f}")

        lines.append("# HELP exo_world_size Number of nodes in cluster")
        lines.append("# TYPE exo_world_size gauge")
        lines.append(f"exo_world_size {self.world_size.get():.0f}")

        lines.append("# HELP exo_errors_total Total inference errors")
        lines.append("# TYPE exo_errors_total counter")
        lines.append(f"exo_errors_total {self.errors_total.get():.0f}")

        lines.append(
            "# HELP exo_memory_pressure_ratio Current RAM pressure ratio (0.0-1.0)"
        )
        lines.append("# TYPE exo_memory_pressure_ratio gauge")
        lines.append(
            f"exo_memory_pressure_ratio {self.memory_pressure_ratio.get():.4f}"
        )

        lines.append(
            "# HELP exo_circuit_breaker_open 1 if circuit breaker is OPEN per model, else 0"
        )
        lines.append("# TYPE exo_circuit_breaker_open gauge")
        lines.append(
            f'exo_circuit_breaker_open{{model="default"}} {self.circuit_breaker_open.get():.0f}'
        )

        lines.append(
            "# HELP exo_priority_queue_depth Current number of requests waiting in the priority queue"
        )
        lines.append("# TYPE exo_priority_queue_depth gauge")
        lines.append(f"exo_priority_queue_depth {self.priority_queue_depth.get():.0f}")

        lines.append(
            "# HELP exo_rate_limit_rejected_total Cumulative requests rejected by the rate limiter"
        )
        lines.append("# TYPE exo_rate_limit_rejected_total counter")
        lines.append(
            f"exo_rate_limit_rejected_total {self.rate_limit_rejected_total.get():.0f}"
        )

        lines.append(
            "# HELP exo_inference_latency_p50_ms Rolling p50 inference latency in milliseconds"
        )
        lines.append("# TYPE exo_inference_latency_p50_ms gauge")
        lines.append(
            f"exo_inference_latency_p50_ms {self.inference_latency_p50_ms.get():.3f}"
        )

        lines.append(
            "# HELP exo_inference_latency_p99_ms Rolling p99 inference latency in milliseconds"
        )
        lines.append("# TYPE exo_inference_latency_p99_ms gauge")
        lines.append(
            f"exo_inference_latency_p99_ms {self.inference_latency_p99_ms.get():.3f}"
        )

        lines.append(
            "# HELP exo_load_shed_total Cumulative requests shed due to overload"
        )
        lines.append("# TYPE exo_load_shed_total counter")
        lines.append(f"exo_load_shed_total {self.load_shed_total.get():.0f}")

        lines.append("# HELP exo_kv_cache_hit_rate KV cache hit rate (0.0-1.0)")
        lines.append("# TYPE exo_kv_cache_hit_rate gauge")
        lines.append(f"exo_kv_cache_hit_rate {self.kv_cache_hit_rate.get():.4f}")

        lines.append("# HELP exo_memory_pressure Current RAM pressure (0.0-1.0)")
        lines.append("# TYPE exo_memory_pressure gauge")
        lines.append(f"exo_memory_pressure {self.memory_pressure.get():.4f}")

        lines.append(
            "# HELP exo_error_budget_remaining_pct Percentage of error budget not yet consumed"
        )
        lines.append("# TYPE exo_error_budget_remaining_pct gauge")
        lines.append(
            f"exo_error_budget_remaining_pct {self.error_budget_remaining_pct.get():.4f}"
        )

        lines.append(
            "# HELP exo_slo_violation_rate Aggregate SLO violation rate (0.0-1.0)"
        )
        lines.append("# TYPE exo_slo_violation_rate gauge")
        lines.append(f"exo_slo_violation_rate {self.slo_violation_rate.get():.4f}")

        counts, hsum, htotal = self.request_duration_seconds.get()
        lines.append("# HELP exo_request_duration_seconds Inference request duration")
        lines.append("# TYPE exo_request_duration_seconds histogram")
        for bucket, count in sorted(counts.items()):
            lines.append(
                f'exo_request_duration_seconds_bucket{{le="{bucket}"}} {count}'
            )
        lines.append(f'exo_request_duration_seconds_bucket{{le="+Inf"}} {htotal}')
        lines.append(f"exo_request_duration_seconds_sum {hsum:.6f}")
        lines.append(f"exo_request_duration_seconds_count {htotal}")

        return "\n".join(lines) + "\n"


METRICS = Metrics()
