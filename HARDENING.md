# Distributed-Systems Hardening — Production Fork

33 hardening commits on top of upstream `main` (07598a3a). All code tested: **266 passing behavioral tests** (188 added here), `ruff` clean.

## Engine fixes
- **Synchronous placement validation** (`api`): `POST /place_instance` dry-runs placement against the API's state replica and returns `400` for requests the topology can never satisfy — previously accepted silently and failed asynchronously with no client signal.
- **Error surfacing** (`master`): placement/inference `ValueError`s surface as `ErrorChunk` so streams terminate instead of hanging.
- **`EXO_LIBP2P_PORT`**: pin the libp2p listen port via env (default unchanged: OS-assigned).
- Swallowed-exception logging in download scan + cache tier.

## Ops primitives library (`src/exo/master/`)
55 standalone modules, each committed atomically with behavioral tests:

| Domain | Modules |
|---|---|
| Resilience | circuit breakers (worker/model), failover, reconnect backoff, graceful degradation |
| Traffic | admission control, load shedding, rate limiting (token-bucket/distributed/SLO-aware), request hedging, retry policy, adaptive timeouts |
| Observability | SLO tracking, burn-rate error budgets, composite health scoring, anomaly detection, profilers, prometheus aggregation |
| Placement | shard placement/rebalancing/validation, topology graph (path scoring, cycle enumeration) |
| Efficiency | response/result caching, request dedup by fingerprint, batched inference, stream compression |
| Infrastructure | heartbeat monitoring, node registry, event stream, connection pooling, config hot-reload, runbook automation |

## Bug fixes (caught by the new test suites)
- Rate limiter: `defaultdict` factory hardcoded authenticated RPM — anonymous clients silently received 6× intended capacity on direct subscript.
- Request hedging: never-awaited hedge coroutine leaked (`RuntimeWarning`) when the primary won during the hedge delay.
- Inference profiler: late-binding closure over loop variables in percentile computation.
- Model registry: missing logger import (latent `NameError` on registry parse failure).

## Test run
```
PYTHONPATH=src pytest src/exo/master/tests/ src/exo/api/tests/
266 passed
```
