"""
AIOps Platform -- Full Telemetry Data Generator (20k rows, 13 failure modes)
=============================================================================
Generates synthetic telemetry for all 13 failure modes defined in the HMM pipeline:

  1  NONE               - Healthy baseline
  2  MEMORY_LEAK        - Gradual heap growth, rising GC pressure
  3  CPU_SATURATION     - CPU pegged, all latencies elevated uniformly
  4  LATENCY_SPIKE      - Intermittent GC-induced P99 spikes, P50 flat
  5  ERROR_STORM        - Error rate surge, latency secondary
  6  DB_SLOWDOWN        - DB P99 slowly climbs, cache miss rate stays normal
  7  CACHE_STAMPEDE     - Sudden cache miss surge -> DB overload (M/M/1)
  8  QUEUE_BACKUP       - Queue lag balloons, throughput plateau
  9  DEPENDENCY_TIMEOUT - Downstream call timeouts, circuit-breaker trips
 10  BAD_DEPLOY         - Immediate step-change error rate at canary rollout
 11  RETRY_STORM        - Retry rate explodes, effective RPS 2-4x
 12  DISK_IO_SATURATION - Disk-related DB slowdown, CPU low
 13  CASCADING_FAILURE  - Multi-metric surge: errors + latency + CPU

Output (appended or fresh):
  data/telemetry_metrics.csv   -- one metric row per 2s step
  data/telemetry_logs.csv      -- log line per step
  data/telemetry_traces.jsonl  -- one trace per episode per ~4 steps

All values ASCII-only (Windows cp1252 safe).
"""

import argparse
import csv
import json
import math
import os
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ALL_MODES = [
    "NONE",
    "MEMORY_LEAK",
    "CPU_SATURATION",
    "LATENCY_SPIKE",
    "ERROR_STORM",
    "DB_SLOWDOWN",
    "CACHE_STAMPEDE",
    "QUEUE_BACKUP",
    "DEPENDENCY_TIMEOUT",
    "BAD_DEPLOY",
    "RETRY_STORM",
    "DISK_IO_SATURATION",
    "CASCADING_FAILURE",
]

SERVICES = ["auth-service", "payment-service", "order-service", "inventory-service"]


@dataclass
class Config:
    seed: int = 42
    # Episodes per failure mode -- tuned so total metric rows ~ 20000
    # 13 modes * 12 episodes * 120 steps/episode = 18720 (close to 20k)
    # We use 13 modes * 13 episodes * 120 steps = 20280
    episodes_per_mode: int = 13
    steps_per_episode: int = 120      # 120 * 2s = 4 minutes
    step_interval_s: int = 2
    output_dir: str = "data"
    base_rps: float = 200.0
    db_capacity_rps: float = 250.0


# ---------------------------------------------------------------------------
# Statistical distributions
# ---------------------------------------------------------------------------

class Dist:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    # ---- Baseline -----------------------------------------------------------
    def p50(self): return float(np.clip(self.rng.normal(90, 8), 50, 200))
    def p99(self): return float(np.clip(self.rng.normal(95, 10), 60, 250))
    def heap(self): return float(np.clip(self.rng.normal(512, 20), 400, 700))
    def gc_pause(self): return float(np.clip(self.rng.exponential(12), 2, 40))
    def err_rate(self): return float(self.rng.beta(2, 198))
    def cache_miss(self): return float(np.clip(self.rng.beta(1, 19), 0.01, 0.12))
    def db_p99(self): return float(np.clip(self.rng.lognormal(3.0, 0.3), 5, 80))
    def cpu(self): return float(np.clip(self.rng.normal(35, 5), 15, 55))
    def queue_lag(self): return int(self.rng.integers(0, 15))
    def retry_rate(self): return float(np.clip(self.rng.beta(1, 49), 0.0, 0.05))
    def rps(self): return float(np.clip(self.rng.normal(200, 20), 80, 400))

    # ---- MEMORY_LEAK --------------------------------------------------------
    def heap_leak(self, elapsed_s: int, rate_mb_per_min: float = 12.0) -> float:
        """Heap grows linearly at rate_mb_per_min, capped at 1800MB."""
        growth = (elapsed_s / 60.0) * rate_mb_per_min
        noise = float(self.rng.normal(0, 8))
        return float(np.clip(512 + growth + noise, 400, 1800))

    def gc_leak(self, heap_mb: float) -> float:
        """GC pause scales super-linearly with heap fill."""
        ratio = max(0, (heap_mb - 512) / (1800 - 512))
        return float(np.clip(self.rng.lognormal(math.log(15 + ratio * 400), 0.3), 2, 800))

    # ---- CPU_SATURATION -----------------------------------------------------
    def cpu_sat(self): return float(np.clip(self.rng.normal(88, 4), 70, 100))
    def p99_cpu_sat(self, cpu: float) -> float:
        return float(np.clip(self.rng.normal(200 + (cpu - 70) * 8, 30), 150, 1500))

    # ---- LATENCY_SPIKE ------------------------------------------------------
    def gc_spike_ms(self): return float(np.clip(self.rng.lognormal(5.5, 0.4), 80, 650))
    def p99_spike(self, gc_ms: float) -> float:
        return float(np.clip(gc_ms * float(self.rng.uniform(0.8, 1.1)) + 90 + float(self.rng.normal(0, 30)), 200, 3000))
    def gc_event(self, lam: float = 0.08) -> bool:
        return self.rng.poisson(lam) > 0

    # ---- ERROR_STORM --------------------------------------------------------
    def err_storm(self): return float(np.clip(self.rng.beta(8, 12), 0.25, 0.70))
    def p99_error_storm(self): return float(np.clip(self.rng.normal(130, 20), 80, 300))

    # ---- DB_SLOWDOWN --------------------------------------------------------
    def db_slow(self, elapsed_s: int) -> float:
        """DB P99 grows slowly -- LogNormal drift."""
        growth = (elapsed_s / 240.0) * 300
        return float(np.clip(self.rng.lognormal(math.log(max(20, 20 + growth)), 0.25), 20, 1500))

    # ---- CACHE_STAMPEDE -----------------------------------------------------
    def cache_miss_stampede(self) -> float:
        return float(np.clip(self.rng.beta(19, 1) * 0.97 + self.rng.normal(0, 0.025), 0.85, 0.99))

    def db_p99_mm1(self, miss_rate: float, base_rps: float, cap_rps: float) -> float:
        svc_ms = 20.0 * float(self.rng.lognormal(0, 0.05))
        rho = min(base_rps * miss_rate / cap_rps, 0.98)
        queue_ms = svc_ms * (rho / (1 - rho)) if rho < 0.99 else svc_ms * 50
        return float(np.clip(svc_ms + queue_ms + self.rng.normal(0, 8), 20, 1200))

    # ---- QUEUE_BACKUP -------------------------------------------------------
    def queue_lag_backup(self, elapsed_s: int) -> int:
        return int(min(500, int(elapsed_s * 1.8 + self.rng.integers(0, 30))))

    def p99_queue(self, lag: int) -> float:
        return float(np.clip(self.rng.normal(90 + lag * 1.5, 20), 80, 2000))

    # ---- DEPENDENCY_TIMEOUT -------------------------------------------------
    def timeout_rate(self) -> float:
        return float(np.clip(self.rng.beta(5, 5), 0.15, 0.55))

    def p99_timeout(self, rate: float) -> float:
        return float(np.clip(self.rng.normal(300 + rate * 1000, 80), 200, 3000))

    # ---- BAD_DEPLOY ---------------------------------------------------------
    def err_deploy(self) -> float:
        return float(np.clip(self.rng.beta(12, 8), 0.30, 0.80))

    # ---- RETRY_STORM --------------------------------------------------------
    def retry_high(self) -> float:
        return float(np.clip(self.rng.beta(10, 5), 0.40, 0.90))

    def rps_retry(self, retry_rate: float, base_rps: float) -> float:
        return float(np.clip(base_rps * (1 + retry_rate * 3), base_rps, base_rps * 4))

    # ---- DISK_IO_SATURATION -------------------------------------------------
    def db_slow_disk(self, elapsed_s: int) -> float:
        growth = (elapsed_s / 240.0) * 500
        return float(np.clip(self.rng.lognormal(math.log(max(40, 40 + growth)), 0.3), 40, 2000))

    def cpu_disk_sat(self) -> float:
        # CPU low because it's waiting on IO, not compute
        return float(np.clip(self.rng.normal(22, 4), 8, 40))

    # ---- CASCADING_FAILURE --------------------------------------------------
    def cascade_cpu(self) -> float:
        return float(np.clip(self.rng.normal(82, 6), 60, 100))

    def cascade_err(self) -> float:
        return float(np.clip(self.rng.beta(10, 6), 0.35, 0.85))

    def cascade_p99(self, cpu: float, err: float) -> float:
        return float(np.clip(self.rng.normal(800 + cpu * 8 + err * 200, 100), 500, 5000))


# ---------------------------------------------------------------------------
# Span / Trace builders
# ---------------------------------------------------------------------------

class Span:
    def __init__(self, trace_id, span_id, parent_id, op, svc, start_ms, dur_ms, status, attrs):
        self.d = {
            "trace_id": trace_id, "span_id": span_id, "parent_span_id": parent_id,
            "operation": op, "service": svc,
            "start_offset_ms": round(start_ms, 2), "duration_ms": round(dur_ms, 2),
            "status": status, "attributes": attrs,
        }


def _trace(service, timestamp, failure_mode, spans_data, extra=None):
    tid = str(uuid.uuid4())
    out = {"trace_id": tid, "timestamp": timestamp, "service": service,
           "failure_mode": failure_mode, "spans": []}
    for op, parent, start, dur, status, attrs in spans_data:
        sid = str(uuid.uuid4())
        s = Span(tid, sid, parent, op, service, start, dur, status, attrs)
        out["spans"].append(s.d)
        if parent is None:
            out["spans"][0]["span_id"] = sid   # root id
    if extra:
        out.update(extra)
    return out


def make_trace(service, timestamp, failure_mode, rng: np.random.Generator):
    """Route to the correct trace builder per failure mode."""
    d = Dist(rng)
    tid = str(uuid.uuid4())
    root_id = str(uuid.uuid4())
    cache_id = str(uuid.uuid4())
    db_id = str(uuid.uuid4())
    dep_id = str(uuid.uuid4())

    if failure_mode == "NONE":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(8, 2), 2, 20))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "MEMORY_LEAK":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        gc_ms = float(np.clip(rng.lognormal(5.0, 0.5), 40, 500))
        db_ms = float(np.clip(rng.normal(18, 4), 8, 40))
        root_ms = gc_ms + cache_ms + db_ms + float(rng.normal(5, 2))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(max(root_ms, 10), 2),
             "status": "ok", "attributes": {"http.method": "GET", "gc.pressure": "high"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "CPU_SATURATION":
        cache_ms = float(np.clip(rng.normal(5, 1), 2, 15))
        db_ms = float(np.clip(rng.normal(25, 5), 10, 60))
        cpu_delay = float(np.clip(rng.normal(150, 40), 50, 500))
        root_ms = cpu_delay + cache_ms + db_ms
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET", "cpu.throttled": True}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": cpu_delay, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cpu_delay + cache_ms, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "LATENCY_SPIKE":
        gc_ms = float(np.clip(rng.lognormal(5.5, 0.4), 80, 650))
        affected = rng.random() < 0.08
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        if affected:
            root_ms = gc_ms + cache_ms + db_ms + float(rng.normal(8, 2))
            root_attrs = {"http.method": "GET", "gc.pause_detected": True, "gc.pause_ms": round(gc_ms, 1)}
        else:
            root_ms = cache_ms + db_ms + float(np.clip(rng.normal(8, 2), 2, 20))
            root_attrs = {"http.method": "GET", "gc.pause_detected": False}
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(max(root_ms, 10), 2),
             "status": "ok", "attributes": root_attrs},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "ERROR_STORM":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(18, 5), 8, 80))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        status = "error" if rng.random() < 0.45 else "ok"
        exception = "RuntimeException" if status == "error" else ""
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": status, "attributes": {"http.method": "GET", "http.status_code": 500 if status == "error" else 200}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": status, "attributes": {"db.type": "postgresql", "db.exception": exception}},
        ]

    elif failure_mode == "DB_SLOWDOWN":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.lognormal(5.5, 0.4), 100, 1500))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql", "db.slow_query": True}},
        ]

    elif failure_mode == "CACHE_STAMPEDE":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 6))
        db_ms = float(np.clip(rng.lognormal(5.2, 0.35), 60, 1200))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 30))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "MISS", "cache.key_type": "user_session"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql", "db.overloaded": True,
                                             "db.rows_examined": int(rng.integers(1000, 50000))}},
        ]

    elif failure_mode == "QUEUE_BACKUP":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        queue_wait = float(np.clip(rng.exponential(200), 50, 1500))
        db_ms = float(np.clip(rng.normal(20, 5), 8, 60))
        root_ms = queue_wait + cache_ms + db_ms
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET", "queue.wait_ms": round(queue_wait, 1)}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": queue_wait, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(queue_wait + cache_ms, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "DEPENDENCY_TIMEOUT":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        timeout_ms = float(np.clip(rng.normal(5000, 500), 2000, 30000))
        db_ms = float(np.clip(rng.normal(15, 3), 5, 35))
        root_ms = timeout_ms + cache_ms
        status = "error" if rng.random() < 0.6 else "ok"
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": status, "attributes": {"http.method": "GET"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": dep_id, "parent_span_id": root_id,
             "operation": "downstream.call", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(timeout_ms, 2),
             "status": "timeout", "attributes": {"peer.service": "downstream-api", "timeout": True}},
        ]

    elif failure_mode == "BAD_DEPLOY":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(15, 5), 5, 50))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 25))
        status = "error" if rng.random() < 0.55 else "ok"
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": status,
             "attributes": {"http.method": "GET", "deploy.version": "canary-v2.1.3",
                            "http.status_code": 500 if status == "error" else 200}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": status, "attributes": {"db.type": "postgresql"}},
        ]

    elif failure_mode == "RETRY_STORM":
        retry_count = int(rng.integers(2, 8))
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        db_ms = float(np.clip(rng.normal(18, 4), 8, 45))
        root_ms = (cache_ms + db_ms) * retry_count + float(rng.normal(10, 5))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET", "retry.count": retry_count}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms * retry_count, 2),
             "status": "ok", "attributes": {"db.type": "postgresql", "db.retries": retry_count}},
        ]

    elif failure_mode == "DISK_IO_SATURATION":
        cache_ms = float(np.clip(rng.normal(3, 0.5), 1, 8))
        disk_wait = float(np.clip(rng.exponential(300), 80, 2000))
        db_ms = disk_wait + float(np.clip(rng.normal(20, 5), 8, 50))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(10, 3), 3, 20))
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": "ok", "attributes": {"http.method": "GET"}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "HIT"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": "ok", "attributes": {"db.type": "postgresql", "disk.io_wait_ms": round(disk_wait, 1)}},
        ]

    else:  # CASCADING_FAILURE
        cache_ms = float(np.clip(rng.normal(5, 1), 2, 15))
        db_ms = float(np.clip(rng.lognormal(6.0, 0.5), 200, 5000))
        root_ms = cache_ms + db_ms + float(np.clip(rng.normal(80, 30), 20, 300))
        status = "error" if rng.random() < 0.65 else "ok"
        spans = [
            {"trace_id": tid, "span_id": root_id, "parent_span_id": None,
             "operation": "handle_request", "service": service,
             "start_offset_ms": 0.0, "duration_ms": round(root_ms, 2),
             "status": status, "attributes": {"http.method": "GET", "cascade.detected": True}},
            {"trace_id": tid, "span_id": cache_id, "parent_span_id": root_id,
             "operation": "cache_lookup", "service": service,
             "start_offset_ms": 0.5, "duration_ms": round(cache_ms, 2),
             "status": "ok", "attributes": {"cache.result": "MISS"}},
            {"trace_id": tid, "span_id": db_id, "parent_span_id": root_id,
             "operation": "db_query", "service": service,
             "start_offset_ms": round(cache_ms + 0.5, 2), "duration_ms": round(db_ms, 2),
             "status": status, "attributes": {"db.type": "postgresql", "db.overloaded": True}},
        ]

    return {"trace_id": tid, "timestamp": timestamp, "service": service,
            "failure_mode": failure_mode, "spans": spans}


# ---------------------------------------------------------------------------
# Episode generators
# ---------------------------------------------------------------------------

LOG_TEMPLATES = {
    "NONE":               "OK cpu={cpu:.0f}% p99={p99:.0f}ms error_rate={err:.1f}%",
    "MEMORY_LEAK":        "WARN heap={heap:.0f}MB gc_pause={gc:.0f}ms -- memory leak suspected",
    "CPU_SATURATION":     "WARN cpu={cpu:.0f}% -- cpu saturated p99={p99:.0f}ms",
    "LATENCY_SPIKE":      "WARN latency spike p99={p99:.0f}ms gc_pause={gc:.0f}ms heap={heap:.0f}MB",
    "ERROR_STORM":        "ERROR error_rate={err:.1f}% p99={p99:.0f}ms -- error storm",
    "DB_SLOWDOWN":        "WARN db_p99={db:.0f}ms -- database slow query detected",
    "CACHE_STAMPEDE":     "WARN cache_miss={miss:.0f}% db_p99={db:.0f}ms -- cache stampede",
    "QUEUE_BACKUP":       "WARN queue_lag={lag}ms p99={p99:.0f}ms -- queue backup growing",
    "DEPENDENCY_TIMEOUT": "ERROR downstream_timeout={dep:.0f}ms -- dependency unavailable",
    "BAD_DEPLOY":         "ERROR bad deploy error_rate={err:.1f}% version=canary-v2.1.3",
    "RETRY_STORM":        "WARN retry_rate={retry:.0f}% effective_rps={rps:.0f} -- retry storm",
    "DISK_IO_SATURATION": "WARN db_p99={db:.0f}ms cpu={cpu:.0f}% -- disk io saturation",
    "CASCADING_FAILURE":  "CRITICAL cascade cpu={cpu:.0f}% error_rate={err:.1f}% p99={p99:.0f}ms",
}

EXCEPTION_MAP = {
    "ERROR_STORM": "RuntimeException",
    "DEPENDENCY_TIMEOUT": "SocketTimeoutException",
    "BAD_DEPLOY": "NullPointerException",
    "CASCADING_FAILURE": "SystemOverloadException",
}

LOG_LEVELS = {
    "NONE": "INFO",
    "MEMORY_LEAK": "WARNING",
    "CPU_SATURATION": "WARNING",
    "LATENCY_SPIKE": "WARNING",
    "ERROR_STORM": "ERROR",
    "DB_SLOWDOWN": "WARNING",
    "CACHE_STAMPEDE": "WARNING",
    "QUEUE_BACKUP": "WARNING",
    "DEPENDENCY_TIMEOUT": "ERROR",
    "BAD_DEPLOY": "ERROR",
    "RETRY_STORM": "WARNING",
    "DISK_IO_SATURATION": "WARNING",
    "CASCADING_FAILURE": "CRITICAL",
}


def generate_episode(
    episode_id: str,
    failure_mode: str,
    config: Config,
    dist: Dist,
    rng: np.random.Generator,
) -> Tuple[List[dict], List[dict], List[dict]]:

    metrics, logs, traces = [], [], []
    service = str(rng.choice(SERVICES))
    base_ts = int(rng.integers(1_700_000_000, 1_700_500_000))

    for step in range(config.steps_per_episode):
        ts = base_ts + step * config.step_interval_s
        elapsed = step * config.step_interval_s

        # ----- compute shared-state signal vector -----
        if failure_mode == "NONE":
            p50_v = dist.p50(); p99_v = dist.p99()
            heap_v = dist.heap(); gc_v = dist.gc_pause()
            err_v = dist.err_rate(); miss_v = dist.cache_miss()
            db_v = dist.db_p99(); cpu_v = dist.cpu()
            lag_v = dist.queue_lag(); retry_v = dist.retry_rate()
            rps_v = dist.rps()
            extra_log = {"cpu": cpu_v, "p99": p99_v, "err": err_v * 100}

        elif failure_mode == "MEMORY_LEAK":
            heap_v = dist.heap_leak(elapsed)
            gc_v   = dist.gc_leak(heap_v)
            p99_v  = float(np.clip(dist.p99() + gc_v * 0.4, 80, 2000))
            p50_v  = dist.p50()
            err_v  = dist.err_rate()
            miss_v = dist.cache_miss()
            db_v   = dist.db_p99()
            cpu_v  = float(np.clip(dist.cpu() + (heap_v - 512) / 80, 15, 90))
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"heap": heap_v, "gc": gc_v, "err": err_v * 100}

        elif failure_mode == "CPU_SATURATION":
            cpu_v  = dist.cpu_sat()
            p99_v  = dist.p99_cpu_sat(cpu_v)
            p50_v  = float(np.clip(dist.p50() + (cpu_v - 35) * 2, 80, 500))
            heap_v = dist.heap(); gc_v = dist.gc_pause()
            err_v  = float(np.clip(dist.err_rate() + (cpu_v - 70) * 0.002, 0, 0.15))
            miss_v = dist.cache_miss(); db_v = dist.db_p99()
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"cpu": cpu_v, "p99": p99_v, "err": err_v * 100}

        elif failure_mode == "LATENCY_SPIKE":
            gc_fired = dist.gc_event(0.08)
            heap_v = dist.heap(); cpu_v = dist.cpu()
            if gc_fired:
                gc_v   = dist.gc_spike_ms()
                p99_v  = dist.p99_spike(gc_v)
                p50_v  = float(np.clip(rng.normal(92, 4), 70, 120))
                err_v  = float(np.clip(dist.err_rate() + rng.uniform(0.01, 0.04), 0, 0.15))
            else:
                gc_v = dist.gc_pause(); p99_v = dist.p99(); p50_v = dist.p50()
                err_v = dist.err_rate()
            miss_v = dist.cache_miss(); db_v = dist.db_p99()
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"p99": p99_v, "gc": gc_v, "heap": heap_v, "err": err_v * 100}

        elif failure_mode == "ERROR_STORM":
            err_v  = dist.err_storm()
            p99_v  = dist.p99_error_storm()
            p50_v  = dist.p50()
            heap_v = dist.heap(); gc_v = dist.gc_pause(); cpu_v = dist.cpu()
            miss_v = dist.cache_miss(); db_v = dist.db_p99()
            lag_v  = int(np.clip(dist.queue_lag() + err_v * 50, 0, 200))
            retry_v = float(np.clip(err_v * 0.5, 0, 0.5)); rps_v = dist.rps()
            extra_log = {"err": err_v * 100, "p99": p99_v}

        elif failure_mode == "DB_SLOWDOWN":
            db_v   = dist.db_slow(elapsed)
            p99_v  = float(np.clip(db_v * 1.2 + float(rng.normal(30, 10)), db_v, db_v * 1.5))
            p50_v  = dist.p50()
            heap_v = dist.heap(); gc_v = dist.gc_pause(); cpu_v = dist.cpu()
            err_v  = float(np.clip(dist.err_rate() + (db_v - 80) * 0.0002, 0, 0.15))
            miss_v = dist.cache_miss()
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"db": db_v}

        elif failure_mode == "CACHE_STAMPEDE":
            miss_v = dist.cache_miss_stampede()
            db_v   = dist.db_p99_mm1(miss_v, config.base_rps, config.db_capacity_rps)
            p99_v  = float(np.clip(db_v + float(rng.normal(40, 10)), db_v * 0.9, db_v * 1.2))
            p50_v  = dist.p50()
            heap_v = dist.heap(); gc_v = dist.gc_pause()
            cpu_v  = float(np.clip(dist.cpu() + rng.uniform(10, 25), 30, 85))
            err_v  = float(np.clip(rng.beta(3, 12), 0.10, 0.35))
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"miss": miss_v * 100, "db": db_v}

        elif failure_mode == "QUEUE_BACKUP":
            lag_v  = dist.queue_lag_backup(elapsed)
            p99_v  = dist.p99_queue(lag_v)
            p50_v  = float(np.clip(dist.p50() + lag_v * 0.3, 80, 500))
            heap_v = dist.heap(); gc_v = dist.gc_pause(); cpu_v = dist.cpu()
            err_v  = dist.err_rate(); miss_v = dist.cache_miss(); db_v = dist.db_p99()
            retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"lag": lag_v, "p99": p99_v}

        elif failure_mode == "DEPENDENCY_TIMEOUT":
            dep_timeout = dist.timeout_rate()
            p99_v  = dist.p99_timeout(dep_timeout)
            p50_v  = dist.p50()
            err_v  = float(np.clip(dep_timeout * 0.8, 0, 0.65))
            heap_v = dist.heap(); gc_v = dist.gc_pause(); cpu_v = dist.cpu()
            miss_v = dist.cache_miss(); db_v = dist.db_p99()
            lag_v  = dist.queue_lag(); retry_v = float(np.clip(dep_timeout * 0.5, 0, 0.5))
            rps_v  = dist.rps()
            extra_log = {"dep": p99_v, "err": err_v * 100}

        elif failure_mode == "BAD_DEPLOY":
            err_v  = dist.err_deploy()
            p99_v  = float(np.clip(dist.p99() + err_v * 100, 90, 500))
            p50_v  = dist.p50()
            heap_v = dist.heap(); gc_v = dist.gc_pause(); cpu_v = dist.cpu()
            miss_v = dist.cache_miss(); db_v = dist.db_p99()
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"err": err_v * 100}

        elif failure_mode == "RETRY_STORM":
            retry_v = dist.retry_high()
            rps_v   = dist.rps_retry(retry_v, config.base_rps)
            p99_v   = float(np.clip(dist.p99() + retry_v * 200, 90, 1000))
            p50_v   = dist.p50()
            err_v   = dist.err_rate()
            heap_v  = float(np.clip(dist.heap() + retry_v * 50, 400, 800))
            gc_v    = dist.gc_pause(); cpu_v = float(np.clip(dist.cpu() + retry_v * 30, 20, 90))
            miss_v  = dist.cache_miss(); db_v = dist.db_p99()
            lag_v   = int(np.clip(retry_v * 100, 0, 200))
            extra_log = {"retry": retry_v * 100, "rps": rps_v}

        elif failure_mode == "DISK_IO_SATURATION":
            db_v   = dist.db_slow_disk(elapsed)
            cpu_v  = dist.cpu_disk_sat()
            p99_v  = float(np.clip(db_v * 1.1 + float(rng.normal(20, 10)), db_v * 0.8, db_v * 1.5))
            p50_v  = float(np.clip(dist.p50() + db_v * 0.1, 80, 500))
            heap_v = dist.heap(); gc_v = dist.gc_pause()
            err_v  = dist.err_rate(); miss_v = dist.cache_miss()
            lag_v  = dist.queue_lag(); retry_v = dist.retry_rate(); rps_v = dist.rps()
            extra_log = {"db": db_v, "cpu": cpu_v}

        else:  # CASCADING_FAILURE
            cpu_v  = dist.cascade_cpu()
            err_v  = dist.cascade_err()
            p99_v  = dist.cascade_p99(cpu_v, err_v)
            p50_v  = float(np.clip(dist.p50() + cpu_v * 3, 100, 1000))
            heap_v = float(np.clip(dist.heap() + cpu_v * 2, 400, 1000))
            gc_v   = float(np.clip(dist.gc_pause() + cpu_v * 2, 10, 300))
            miss_v = float(np.clip(dist.cache_miss() + err_v * 0.4, 0.05, 0.80))
            db_v   = float(np.clip(dist.db_p99() + p99_v * 0.3, 50, 3000))
            lag_v  = int(np.clip(cpu_v * 3, 50, 500))
            retry_v = float(np.clip(err_v * 0.6, 0, 0.6)); rps_v = dist.rps()
            extra_log = {"cpu": cpu_v, "err": err_v * 100, "p99": p99_v}

        # ----- build metric row -----
        metric_row = {
            "episode_id": episode_id,
            "failure_mode": failure_mode,
            "service": service,
            "elapsed_s": elapsed,
            "timestamp": ts,
            "p50_latency_ms": round(p50_v, 1),
            "p99_latency_ms": round(p99_v, 1),
            "heap_mb": round(heap_v, 1),
            "gc_pause_ms": round(gc_v, 2),
            "error_rate": round(err_v, 4),
            "cache_miss_rate": round(miss_v, 4),
            "db_p99_ms": round(db_v, 1),
            "cpu_pct": round(cpu_v, 1),
            "queue_lag": lag_v,
            "retry_rate": round(retry_v, 4),
            "rps": round(rps_v, 2),
        }
        metrics.append(metric_row)

        # ----- build log row -----
        tmpl = LOG_TEMPLATES[failure_mode]
        try:
            msg = tmpl.format(**extra_log)
        except KeyError:
            msg = f"{failure_mode} detected"

        log_row = {
            "episode_id": episode_id,
            "failure_mode": failure_mode,
            "service": service,
            "elapsed_s": elapsed,
            "timestamp": ts,
            "log_level": LOG_LEVELS[failure_mode],
            "exception_type": EXCEPTION_MAP.get(failure_mode, ""),
            "log_message": msg,
        }
        logs.append(log_row)

        # ----- build trace (every 4 steps) -----
        if step % 4 == 0:
            traces.append(make_trace(service, ts, failure_mode, rng))

    return metrics, logs, traces


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

METRIC_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "p50_latency_ms", "p99_latency_ms", "heap_mb", "gc_pause_ms",
    "error_rate", "cache_miss_rate", "db_p99_ms", "cpu_pct",
    "queue_lag", "retry_rate", "rps",
]

LOG_FIELDS = [
    "episode_id", "failure_mode", "service", "elapsed_s", "timestamp",
    "log_level", "exception_type", "log_message",
]


def write_csv(rows, path, fields, mode="w"):
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    print("  -> Wrote {} rows to {}".format(len(rows), path))


def write_jsonl(records, path, mode="w"):
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    print("  -> Wrote {} records to {}".format(len(records), path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate 20k synthetic telemetry rows")
    parser.add_argument("--episodes", type=int, default=13,
                        help="Episodes per failure mode (default 13 -> ~20k metric rows)")
    parser.add_argument("--steps", type=int, default=120,
                        help="Steps per episode (default 120 = 4 min)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing output files (default: append)")
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        episodes_per_mode=args.episodes,
        steps_per_episode=args.steps,
        output_dir=args.output,
    )

    rng = np.random.default_rng(config.seed)
    dist = Dist(rng)
    os.makedirs(config.output_dir, exist_ok=True)

    metric_path = os.path.join(config.output_dir, "telemetry_metrics.csv")
    log_path    = os.path.join(config.output_dir, "telemetry_logs.csv")
    trace_path  = os.path.join(config.output_dir, "telemetry_traces.jsonl")

    file_mode = "w" if args.overwrite else "a"
    first_write = args.overwrite or not os.path.exists(metric_path)

    print("=" * 65)
    print("AIOps Full Telemetry Generator -- 13 Failure Modes")
    print("  Seed: {}  Episodes/mode: {}  Steps: {}".format(
        config.seed, config.episodes_per_mode, config.steps_per_episode))
    expected = len(ALL_MODES) * config.episodes_per_mode * config.steps_per_episode
    print("  Expected metric rows: ~{}".format(expected))
    print("=" * 65)

    global_ep = 0
    mode_counts = {}

    for mode in ALL_MODES:
        print("\n[{}] Generating {} ({} episodes)...".format(
            ALL_MODES.index(mode) + 1, mode, config.episodes_per_mode))

        mode_metrics, mode_logs, mode_traces = [], [], []

        for i in range(config.episodes_per_mode):
            ep_id = "ep_{:05d}_{}".format(global_ep, mode)
            global_ep += 1
            m, l, t = generate_episode(ep_id, mode, config, dist, rng)
            mode_metrics.extend(m)
            mode_logs.extend(l)
            mode_traces.extend(t)

        mode_counts[mode] = len(mode_metrics)

        # Write mode-by-mode to avoid huge RAM usage
        write_mode = "w" if (first_write and ALL_MODES.index(mode) == 0) else "a"
        write_with_header = (write_mode == "w")

        with open(metric_path, write_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
            if write_with_header:
                writer.writeheader()
            writer.writerows(mode_metrics)

        with open(log_path, write_mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
            if write_with_header:
                writer.writeheader()
            writer.writerows(mode_logs)

        with open(trace_path, write_mode, encoding="utf-8") as f:
            for rec in mode_traces:
                f.write(json.dumps(rec) + "\n")

        print("    {} metric rows, {} log rows, {} traces".format(
            len(mode_metrics), len(mode_logs), len(mode_traces)))

    # Summary
    total = sum(mode_counts.values())
    print("\n" + "=" * 65)
    print("Generation complete. Row counts by failure mode:")
    for mode, cnt in mode_counts.items():
        pct = cnt / total * 100
        print("  {:<22}: {:>5} rows ({:.1f}%)".format(mode, cnt, pct))
    print("\n  TOTAL metric rows : {}".format(total))
    print("  Output directory  : {}".format(os.path.abspath(config.output_dir)))
    print("=" * 65)


if __name__ == "__main__":
    main()
