"""
Prometheus metrics for the field delineation pipeline.

Provides counters, histograms, and gauges for monitoring inference
performance in production. Metrics can be exposed via a push gateway
for batch jobs, or scraped directly for long-running services.

Usage:
    from src.monitoring.metrics import metrics
    metrics.tiles_processed.inc()
    with metrics.inference_latency.time():
        result = model.predict(tile)
"""

import os
import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        push_to_gateway,
    )
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    logger.debug("prometheus_client not installed — metrics disabled")


class PipelineMetrics:
    """Prometheus metrics for the inference pipeline."""

    def __init__(self, enable: bool = True, pushgateway_url: Optional[str] = None):
        self.enabled = enable and HAS_PROMETHEUS
        self.pushgateway_url = pushgateway_url
        self._start_time = time.time()

        if not self.enabled:
            return

        self.registry = CollectorRegistry()

        # ── Counters ─────────────────────────────────────────
        self.tiles_processed = Counter(
            "field_tiles_processed_total",
            "Total tiles processed",
            registry=self.registry,
        )
        self.polygons_created = Counter(
            "field_polygons_created_total",
            "Total polygons created",
            registry=self.registry,
        )
        self.regions_completed = Counter(
            "field_regions_completed_total",
            "Processing regions completed",
            registry=self.registry,
        )
        self.errors_total = Counter(
            "field_errors_total",
            "Total errors encountered",
            ["error_type"],
            registry=self.registry,
        )

        # ── Histograms ───────────────────────────────────────
        self.inference_latency = Histogram(
            "field_inference_latency_seconds",
            "Inference latency per batch",
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )
        self.tile_load_latency = Histogram(
            "field_tile_load_latency_seconds",
            "Time to load a tile batch from disk",
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )
        self.region_latency = Histogram(
            "field_region_latency_seconds",
            "Total time per processing region",
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry,
        )

        # ── Gauges ───────────────────────────────────────────
        self.current_region = Gauge(
            "field_current_region",
            "Current region being processed",
            registry=self.registry,
        )
        self.gpu_memory_mb = Gauge(
            "field_gpu_memory_used_mb",
            "GPU memory used in MB",
            registry=self.registry,
        )
        self.pipeline_uptime = Gauge(
            "field_pipeline_uptime_seconds",
            "Pipeline uptime in seconds",
            registry=self.registry,
        )

    @contextmanager
    def time_inference(self):
        """Context manager to measure inference latency."""
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        self.inference_latency.observe(time.time() - start)

    @contextmanager
    def time_tile_load(self):
        """Context manager to measure tile load latency."""
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        self.tile_load_latency.observe(time.time() - start)

    @contextmanager
    def time_region(self):
        """Context manager to measure region processing time."""
        if not self.enabled:
            yield
            return
        start = time.time()
        yield
        self.region_latency.observe(time.time() - start)

    def record_error(self, error_type: str):
        """Increment error counter."""
        if self.enabled:
            self.errors_total.labels(error_type=error_type).inc()

    def update_gpu_memory(self):
        """Update GPU memory gauge."""
        if not self.enabled:
            return
        try:
            import torch
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() / (1024 * 1024)
                self.gpu_memory_mb.set(mem)
        except Exception:
            pass

    def push(self, job_name: str = "field-delineation"):
        """Push metrics to Prometheus Pushgateway (for batch jobs)."""
        if not self.enabled or not self.pushgateway_url:
            return
        try:
            self.pipeline_uptime.set(time.time() - self._start_time)
            push_to_gateway(self.pushgateway_url, job=job_name, registry=self.registry)
            logger.debug(f"Metrics pushed to {self.pushgateway_url}")
        except Exception as e:
            logger.warning(f"Failed to push metrics: {e}")

    def summary(self) -> dict:
        """Return a summary dict of key metrics (for logging)."""
        if not self.enabled:
            return {}
        return {
            "tiles_processed": self.tiles_processed._value.get(),
            "polygons_created": self.polygons_created._value.get(),
            "regions_completed": self.regions_completed._value.get(),
            "uptime_seconds": round(time.time() - self._start_time, 1),
        }


# ── Global singleton ────────────────────────────────────────
metrics = PipelineMetrics(
    enable=os.environ.get("FIELD_METRICS", "true").lower() == "true",
    pushgateway_url=os.environ.get("PROMETHEUS_PUSHGATEWAY"),
)
