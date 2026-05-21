"""
Health checks for the field delineation pipeline.

Provides liveness and readiness probes for Docker/Kubernetes,
plus diagnostic utilities for GPU, disk, and model status.

Usage:
    from src.monitoring.health import HealthChecker
    hc = HealthChecker()
    status = hc.full_check()
    print(status)  # {"healthy": True, "gpu": {...}, "disk": {...}}
"""

import os
import logging
import shutil
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class HealthChecker:
    """System health checks for production monitoring."""

    def __init__(self, output_dir: str = "data/output", min_disk_gb: float = 1.0):
        self.output_dir = output_dir
        self.min_disk_gb = min_disk_gb

    def check_gpu(self) -> Dict:
        """Check GPU availability and memory."""
        try:
            import torch
            if not torch.cuda.is_available():
                return {"available": False, "reason": "CUDA not available"}

            device_count = torch.cuda.device_count()
            devices = []
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                mem_total = props.total_mem / (1024 ** 3)
                mem_used = torch.cuda.memory_allocated(i) / (1024 ** 3)
                devices.append({
                    "id": i,
                    "name": props.name,
                    "memory_total_gb": round(mem_total, 2),
                    "memory_used_gb": round(mem_used, 2),
                    "memory_free_gb": round(mem_total - mem_used, 2),
                    "utilization_pct": round(100 * mem_used / mem_total, 1),
                })

            return {"available": True, "device_count": device_count, "devices": devices}
        except ImportError:
            return {"available": False, "reason": "torch not installed"}
        except Exception as e:
            return {"available": False, "reason": str(e)}

    def check_disk(self) -> Dict:
        """Check available disk space for output."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            usage = shutil.disk_usage(self.output_dir)
            free_gb = usage.free / (1024 ** 3)
            return {
                "healthy": free_gb >= self.min_disk_gb,
                "free_gb": round(free_gb, 2),
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "min_required_gb": self.min_disk_gb,
            }
        except Exception as e:
            return {"healthy": False, "error": str(e)}

    def check_model(self, model_path: Optional[str] = None) -> Dict:
        """Check if model weights file exists and is readable."""
        if model_path is None:
            return {"status": "skipped", "reason": "no path provided"}
        exists = os.path.isfile(model_path)
        size_mb = os.path.getsize(model_path) / (1024 ** 2) if exists else 0
        return {
            "healthy": exists,
            "path": model_path,
            "size_mb": round(size_mb, 1),
        }

    def check_system(self) -> Dict:
        """Check system resources (CPU, RAM)."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "cpu_count": psutil.cpu_count(),
                "ram_total_gb": round(mem.total / (1024 ** 3), 1),
                "ram_available_gb": round(mem.available / (1024 ** 3), 1),
                "ram_used_pct": mem.percent,
            }
        except ImportError:
            return {"status": "psutil not installed"}

    def full_check(self, model_path: Optional[str] = None) -> Dict:
        """Run all health checks."""
        gpu = self.check_gpu()
        disk = self.check_disk()
        model = self.check_model(model_path)
        system = self.check_system()

        healthy = (
            disk.get("healthy", False)
            and model.get("healthy", True)  # ok if skipped
        )

        return {
            "healthy": healthy,
            "gpu": gpu,
            "disk": disk,
            "model": model,
            "system": system,
        }


def check():
    """Quick health check for Docker HEALTHCHECK."""
    hc = HealthChecker()
    result = hc.full_check()
    if not result["healthy"]:
        raise SystemExit(1)
