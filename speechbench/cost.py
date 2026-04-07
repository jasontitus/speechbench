"""Static cost / runtime estimator.

The orchestrator calls into this BEFORE launching VMs to print:

    "X jobs across N VMs ≈ M wall-hours, ~$Y spot total — proceed?"

Numbers are deliberately rough — we use published RTFx ranges plus a generous
safety multiplier so the estimate errs on the high side.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .config import GPUS, GPUSpec
from .datasets import DATASETS, DatasetSpec
from .models import MODELS, ModelSpec


# Rough average clip length per dataset (seconds). Based on the test split
# averages from the HF dataset cards / Open ASR Leaderboard.
DATASET_AVG_CLIP_S: dict[str, float] = {
    "librispeech_clean": 7.5,
    "librispeech_other": 7.4,
    "ami_ihm": 6.0,        # IHM clips are short utterances on average
    "earnings22": 18.0,    # chunked at ~20s
    "tedlium_longform": 850.0,  # whole-talk merges
    "voxpopuli_en": 8.0,
    "gigaspeech_l": 5.5,
    "spgispeech": 9.0,
}


# Safety multiplier on top of (sec_per_audio_sec * audio_seconds + load).
SAFETY_FACTOR = 1.4


@dataclass
class JobEstimate:
    job_id: str
    model_key: str
    dataset_key: str
    sample_cap: int
    est_audio_seconds: float
    est_wall_seconds: float


@dataclass
class RunEstimate:
    jobs: list[JobEstimate]
    gpu: GPUSpec
    max_vms: int
    total_wall_seconds_serial: float  # if 1 VM did everything
    wall_seconds_per_vm: float  # after partition into max_vms
    total_vm_hours: float
    spot_usd: float
    on_demand_usd: float

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def wall_hours_per_vm(self) -> float:
        return self.wall_seconds_per_vm / 3600.0


def estimate_job(model: ModelSpec, dataset: DatasetSpec, sample_cap: int) -> JobEstimate:
    n_clips = sample_cap if sample_cap > 0 else (dataset.full_size or sample_cap)
    avg_clip_s = DATASET_AVG_CLIP_S.get(dataset.key, 8.0)
    audio_seconds = n_clips * avg_clip_s
    inference_seconds = model.sec_per_audio_sec * audio_seconds
    wall_seconds = (model.load_seconds + inference_seconds) * SAFETY_FACTOR
    from .orchestrate import job_id_for  # local to avoid cycle on import

    return JobEstimate(
        job_id=job_id_for(model.key, dataset.key, sample_cap),
        model_key=model.key,
        dataset_key=dataset.key,
        sample_cap=sample_cap,
        est_audio_seconds=audio_seconds,
        est_wall_seconds=wall_seconds,
    )


def estimate_run(
    job_specs: list[tuple[str, str, int]],
    gpu_key: str = "t4",
    max_vms: int = 1,
) -> RunEstimate:
    """Estimate wall time + cost for a run.

    `job_specs` is a list of (model_key, dataset_key, sample_cap) tuples.

    The runner groups jobs by model and reuses the loaded weights across
    datasets, so the model load cost is paid once per (model, VM) — not once
    per (model, dataset). This estimator mirrors that behavior.
    """
    gpu = GPUS[gpu_key]
    jobs: list[JobEstimate] = []
    for model_key, dataset_key, cap in job_specs:
        jobs.append(estimate_job(MODELS[model_key], DATASETS[dataset_key], cap))

    # Greedy LPT partition by *model* — every job for a single model lands on
    # the same VM so its load cost is amortized. The unit of partitioning is
    # one model's full bundle of (model × all selected datasets) jobs.
    by_model: dict[str, list[JobEstimate]] = {}
    for je in jobs:
        by_model.setdefault(je.model_key, []).append(je)

    # Build "model bundles" with a single load + per-clip inference. The
    # estimate_job() output already includes the load time (× safety) so we
    # need to subtract the duplicate loads when bundling.
    bundles: list[tuple[str, float]] = []
    for mk, mjes in by_model.items():
        load_s = MODELS[mk].load_seconds * SAFETY_FACTOR
        # Sum the per-job estimates (each includes 1× load) and remove the
        # duplicate load contributions, leaving (1 × load) + Σ inference.
        total_with_loads = sum(j.est_wall_seconds for j in mjes)
        bundle_seconds = total_with_loads - (len(mjes) - 1) * load_s
        bundles.append((mk, bundle_seconds))

    # LPT partition the model bundles into bins.
    bins_load = [0.0] * max(1, max_vms)
    for _mk, bs in sorted(bundles, key=lambda b: -b[1]):
        i = bins_load.index(min(bins_load))
        bins_load[i] += bs

    wall_per_vm = max(bins_load)
    total_vm_seconds = sum(bins_load)
    total_vm_hours = total_vm_seconds / 3600.0

    spot_total = total_vm_hours * gpu.spot_usd_per_hr
    on_demand_total = total_vm_hours * gpu.on_demand_usd_per_hr

    return RunEstimate(
        jobs=jobs,
        gpu=gpu,
        max_vms=max(1, max_vms),
        total_wall_seconds_serial=total_vm_seconds,
        wall_seconds_per_vm=wall_per_vm,
        total_vm_hours=total_vm_hours,
        spot_usd=spot_total,
        on_demand_usd=on_demand_total,
    )


def format_seconds(s: float) -> str:
    if s < 60:
        return f"{s:.0f}s"
    if s < 3600:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.2f}h"


def render_estimate(estimate: RunEstimate, *, planned_vms: int, model_keys: list[str], dataset_keys: list[str]) -> str:
    lines: list[str] = []
    lines.append("─" * 78)
    lines.append("speechbench run estimate")
    lines.append("─" * 78)
    lines.append(f"  GPU type:           {estimate.gpu.key} ({estimate.gpu.machine_type}, {estimate.gpu.vram_gb} GB)")
    lines.append(f"  Spot price:         ~${estimate.gpu.spot_usd_per_hr:.2f}/hr per VM")
    lines.append(f"  Models in sweep:    {len(model_keys):3d}   ({', '.join(model_keys[:6])}{' …' if len(model_keys) > 6 else ''})")
    lines.append(f"  Datasets in sweep:  {len(dataset_keys):3d}   ({', '.join(dataset_keys)})")
    lines.append(f"  Jobs to run:        {estimate.num_jobs}")
    lines.append(f"  VMs:                {planned_vms} (parallel)")
    lines.append("")
    lines.append(f"  Estimated wall time on a single VM:    {format_seconds(estimate.total_wall_seconds_serial)}")
    lines.append(f"  Estimated wall time at {planned_vms:>2} VMs in parallel: {format_seconds(estimate.wall_seconds_per_vm)}")
    lines.append(f"  Total VM-hours:                        {estimate.total_vm_hours:.2f}")
    lines.append("")
    lines.append(f"  Estimated SPOT cost:        ${estimate.spot_usd:7.2f}")
    lines.append(f"  Equivalent on-demand cost:  ${estimate.on_demand_usd:7.2f}  (informational)")
    lines.append("")
    lines.append("  Caveats:")
    lines.append("    • Estimates use static throughput priors × 1.4 safety factor.")
    lines.append("    • Real cost depends on actual spot price + DLVM image install time.")
    lines.append("    • DashScope API model jobs are billed separately by Alibaba.")
    lines.append("─" * 78)
    return "\n".join(lines)
