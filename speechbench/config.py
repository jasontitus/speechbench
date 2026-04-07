"""GCP / orchestrator defaults for speechbench."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

PROJECT_ID = os.environ.get("SPEECHBENCH_PROJECT", "open-testimony")
DEFAULT_BUCKET = os.environ.get(
    "SPEECHBENCH_BUCKET", f"{PROJECT_ID}-speechbench"
)
DEFAULT_REGION = os.environ.get("SPEECHBENCH_REGION", "us-central1")
DEFAULT_ZONE = os.environ.get("SPEECHBENCH_ZONE", "us-central1-a")

# DLVM image (PyTorch 2.7 + CUDA 12.8 on Ubuntu 22.04). Resolved by family.
DLVM_IMAGE_PROJECT = "deeplearning-platform-release"
DLVM_IMAGE_FAMILY = "pytorch-2-7-cu128-ubuntu-2204-nvidia-570"


@dataclass(frozen=True)
class GPUSpec:
    """A single GPU machine flavor."""

    key: str  # short name used on the CLI: t4 / l4 / a100
    accelerator_type: str  # gcloud accelerator name
    quota_metric: str  # regional quota metric name (preemptible)
    machine_type: str  # GCE machine type
    vram_gb: int  # nominal per-GPU VRAM
    on_demand_usd_per_hr: float  # rough cost (GPU + machine), us-central1
    spot_usd_per_hr: float


GPUS: dict[str, GPUSpec] = {
    "t4": GPUSpec(
        key="t4",
        accelerator_type="nvidia-tesla-t4",
        quota_metric="PREEMPTIBLE_NVIDIA_T4_GPUS",
        # n1-standard-8 (30 GB RAM) — needed for mmap-loading models like
        # Gemma 4 E4B which has a 16 GB safetensors file. n1-standard-4
        # (15 GB) OOMs at the mmap step.
        machine_type="n1-standard-8",
        vram_gb=16,
        on_demand_usd_per_hr=0.59,
        spot_usd_per_hr=0.18,
    ),
    "l4": GPUSpec(
        key="l4",
        accelerator_type="nvidia-l4",
        quota_metric="PREEMPTIBLE_NVIDIA_L4_GPUS",
        machine_type="g2-standard-8",
        vram_gb=24,
        on_demand_usd_per_hr=0.85,
        spot_usd_per_hr=0.28,
    ),
    "a100": GPUSpec(
        key="a100",
        accelerator_type="nvidia-tesla-a100",
        quota_metric="PREEMPTIBLE_NVIDIA_A100_GPUS",
        machine_type="a2-highgpu-1g",
        vram_gb=40,
        on_demand_usd_per_hr=3.67,
        spot_usd_per_hr=1.10,
    ),
}


@dataclass
class LaunchConfig:
    project: str = PROJECT_ID
    bucket: str = DEFAULT_BUCKET
    region: str = DEFAULT_REGION
    zone: str = DEFAULT_ZONE
    gpu: str = "t4"  # key into GPUS
    max_vms: int = 1  # default within open-testimony quota
    image_project: str = DLVM_IMAGE_PROJECT
    image_family: str = DLVM_IMAGE_FAMILY
    api_keys: dict[str, str] = field(default_factory=dict)

    @property
    def gpu_spec(self) -> GPUSpec:
        return GPUS[self.gpu]
