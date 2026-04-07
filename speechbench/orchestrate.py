"""Local orchestrator: build job list, plan VMs, launch them.

Run from the laptop. Reads the model + dataset registries, computes the job
matrix, prints a cost estimate, asks for confirmation, packages the source
into a tarball, uploads it to GCS, and creates spot GPU VMs across one or
more regions.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import gcp
from .config import GPUS, LaunchConfig
from .cost import estimate_run, render_estimate
from .datasets import DATASETS
from .models import MODELS


# ─── Job ID + manifest ─────────────────────────────────────────────────────────


def job_id_for(model_key: str, dataset_key: str, sample_cap: int) -> str:
    h = hashlib.sha1(f"{model_key}|{dataset_key}|{sample_cap}".encode()).hexdigest()
    return h[:12]


@dataclass
class Job:
    job_id: str
    model_key: str
    dataset_key: str
    sample_cap: int

    @classmethod
    def from_keys(cls, model_key: str, dataset_key: str, sample_cap: int) -> "Job":
        return cls(
            job_id=job_id_for(model_key, dataset_key, sample_cap),
            model_key=model_key,
            dataset_key=dataset_key,
            sample_cap=sample_cap,
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Multi-region parallelism ──────────────────────────────────────────────────
#
# open-testimony has 1 spot T4 + 1 spot L4 per region in (almost) every GCP
# region without needing a quota raise. To run more than 1 VM in parallel we
# spread VMs across these regions. Each VM uses ITS region's T4 (or L4) quota.

# Priority list — we fan out in this order. Cheap + close first.
PARALLEL_REGION_ORDER = [
    ("us-central1", "us-central1-a"),
    ("us-east1", "us-east1-c"),
    ("us-west1", "us-west1-b"),
    ("us-east4", "us-east4-c"),
    ("us-west4", "us-west4-a"),
    ("europe-west1", "europe-west1-b"),
    ("europe-west4", "europe-west4-a"),
    ("asia-east1", "asia-east1-a"),
    ("asia-southeast1", "asia-southeast1-b"),
]


def regions_with_quota(project: str, gpu_key: str, want_vms: int) -> list[tuple[str, str]]:
    """Return up to `want_vms` (region, zone) pairs that have at least 1 of the
    requested GPU type quota in `project`."""
    spec = GPUS[gpu_key]
    out: list[tuple[str, str]] = []
    for region, zone in PARALLEL_REGION_ORDER:
        if len(out) >= want_vms:
            break
        try:
            limit = gcp.regional_quota(project, region, spec.quota_metric)
        except Exception as e:
            print(f"  ! cannot read quota for {region} ({spec.quota_metric}): {e}", file=sys.stderr)
            continue
        if limit >= 1:
            out.append((region, zone))
    return out


# ─── Source packaging ──────────────────────────────────────────────────────────


def tar_source(repo_root: Path, dest: Path) -> Path:
    """Tar up the source needed by the VM (speechbench/ + requirements-vm.txt).

    Excludes __pycache__/ and *.pyc so we don't ship cached compile output
    from a different Python version than the VM will run.
    """

    def _filter(info: tarfile.TarInfo) -> Optional[tarfile.TarInfo]:
        name = info.name
        if "__pycache__" in name.split("/"):
            return None
        if name.endswith(".pyc"):
            return None
        if name.endswith(".egg-info") or "/.egg-info/" in name:
            return None
        return info

    with tarfile.open(dest, "w:gz") as tf:
        tf.add(repo_root / "speechbench", arcname="speechbench", filter=_filter)
        tf.add(repo_root / "requirements-vm.txt", arcname="requirements-vm.txt")
    return dest


# ─── Job partitioning (greedy LPT) ─────────────────────────────────────────────


def partition_jobs(jobs: list[Job], num_bins: int) -> list[list[Job]]:
    """Greedy LPT partition into roughly equal bins, *bundled by model*.

    The runner amortizes the model load cost across all (model × N datasets)
    jobs that share a model_key, so a single model bundle must land on a
    single VM. We partition at the model-bundle granularity, then flatten the
    chosen bundles into per-bin job lists.
    """
    from .cost import SAFETY_FACTOR, estimate_job

    # Build bundles
    bundles: dict[str, list[Job]] = {}
    bundle_weight: dict[str, float] = {}
    for j in jobs:
        bundles.setdefault(j.model_key, []).append(j)

    for mk, group in bundles.items():
        spec = MODELS[mk]
        # one load + sum of inference times across this model's jobs
        load_s = spec.load_seconds * SAFETY_FACTOR
        per_job_weights = [
            estimate_job(spec, DATASETS[j.dataset_key], j.sample_cap).est_wall_seconds
            for j in group
        ]
        bundle_weight[mk] = load_s + sum(w - load_s for w in per_job_weights)

    # LPT partition the bundles
    weighted_bundles = sorted(bundle_weight.items(), key=lambda kv: -kv[1])
    bins: list[list[Job]] = [[] for _ in range(max(1, num_bins))]
    bin_loads = [0.0] * len(bins)
    for mk, w in weighted_bundles:
        i = bin_loads.index(min(bin_loads))
        bins[i].extend(bundles[mk])
        bin_loads[i] += w
    return bins


# ─── Resume: list completed jobs ───────────────────────────────────────────────


def already_completed(bucket: str, run_id: str) -> set[str]:
    bucket_uri = bucket if bucket.startswith("gs://") else f"gs://{bucket}"
    prefix = f"{bucket_uri}/runs/{run_id}/results/"
    out = subprocess.run(
        ["gsutil", "ls", prefix],
        capture_output=True,
        text=True,
    )
    if out.returncode != 0:
        return set()
    done = set()
    for line in out.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        name = line.rsplit("/", 1)[-1]
        if name.endswith(".json"):
            done.add(name.removesuffix(".json").removesuffix(".failed"))
    return done


# ─── The main launch flow ──────────────────────────────────────────────────────


@dataclass
class LaunchPlan:
    run_id: str
    bucket: str
    src_uri: str
    jobs: list[Job]
    skipped: list[Job]
    bins: list[list[Job]]
    regions: list[tuple[str, str]]  # (region, zone) per VM


def plan_launch(
    *,
    cfg: LaunchConfig,
    model_keys: list[str],
    dataset_keys: list[str],
    sample_caps: dict[str, int],
    run_id: Optional[str],
    rerun: bool,
    multi_region: bool,
) -> LaunchPlan:
    # 1. build full job matrix
    all_jobs: list[Job] = []
    for m in model_keys:
        if m not in MODELS:
            raise SystemExit(f"unknown model: {m}")
        for d in dataset_keys:
            if d not in DATASETS:
                raise SystemExit(f"unknown dataset: {d}")
            cap = sample_caps.get(d, DATASETS[d].default_cap)
            all_jobs.append(Job.from_keys(m, d, cap))

    # 2. resume diff
    rid = run_id or _new_run_id()
    bucket = cfg.bucket
    completed_ids: set[str] = set()
    if not rerun:
        completed_ids = already_completed(bucket, rid)

    todo: list[Job] = [j for j in all_jobs if j.job_id not in completed_ids]
    skipped: list[Job] = [j for j in all_jobs if j.job_id in completed_ids]

    # 3. choose regions / zones
    if multi_region and cfg.max_vms > 1:
        regions = regions_with_quota(cfg.project, cfg.gpu, cfg.max_vms)
        if len(regions) < cfg.max_vms:
            print(
                f"  ! Only found quota for {len(regions)} {cfg.gpu} VMs across {len(PARALLEL_REGION_ORDER)} regions; "
                f"reducing --max-vms from {cfg.max_vms} to {len(regions)}",
                file=sys.stderr,
            )
            cfg.max_vms = max(1, len(regions))
    else:
        regions = [(cfg.region, cfg.zone)] * cfg.max_vms

    # 4. partition todo across max_vms (or fewer if there are fewer jobs)
    n_bins = min(cfg.max_vms, max(1, len(todo)))
    bins = partition_jobs(todo, n_bins)
    # ensure regions list matches bin count
    regions = regions[:n_bins] if regions else [(cfg.region, cfg.zone)] * n_bins
    while len(regions) < n_bins:
        regions.append((cfg.region, cfg.zone))

    return LaunchPlan(
        run_id=rid,
        bucket=bucket,
        src_uri=f"gs://{bucket}/src/{rid}.tar.gz",
        jobs=todo,
        skipped=skipped,
        bins=bins,
        regions=regions,
    )


def _new_run_id() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def upload_assignments(plan: LaunchPlan) -> list[str]:
    """Write per-VM assignments and the master jobs.json to GCS."""
    bucket_uri = f"gs://{plan.bucket}"
    base = f"{bucket_uri}/runs/{plan.run_id}"

    # Master job list
    jobs_payload = {
        "run_id": plan.run_id,
        "jobs": [j.to_dict() for j in plan.jobs],
        "skipped": [j.to_dict() for j in plan.skipped],
    }
    gcp.gs_upload_string(json.dumps(jobs_payload, indent=2), f"{base}/jobs.json")

    assignment_uris: list[str] = []
    for i, bin_jobs in enumerate(plan.bins):
        vm_idx = i + 1
        payload = {
            "vm_index": vm_idx,
            "run_id": plan.run_id,
            "jobs": [j.to_dict() for j in bin_jobs],
        }
        uri = f"{base}/assignments/vm-{vm_idx:02d}.json"
        gcp.gs_upload_string(json.dumps(payload, indent=2), uri)
        assignment_uris.append(uri)
    return assignment_uris


def upload_source(plan: LaunchPlan, repo_root: Path) -> str:
    with tempfile.TemporaryDirectory() as tmpd:
        tar_path = Path(tmpd) / f"src-{plan.run_id}.tar.gz"
        tar_source(repo_root, tar_path)
        gcp.gs_upload(str(tar_path), plan.src_uri)
    return plan.src_uri


def launch_vms(
    plan: LaunchPlan,
    cfg: LaunchConfig,
    assignment_uris: list[str],
    *,
    rerun: bool,
    startup_script_path: Path,
    name_prefix: str = "speechbench",
) -> list[dict]:
    spec = cfg.gpu_spec
    metadata_common = {
        "bucket": f"gs://{plan.bucket}",
        "run-id": plan.run_id,
        "src-uri": plan.src_uri,
        "rerun": "true" if rerun else "false",
    }
    if "DASHSCOPE_API_KEY" in cfg.api_keys:
        metadata_common["dashscope-api-key"] = cfg.api_keys["DASHSCOPE_API_KEY"]

    instances: list[dict] = []
    for i, (assign_uri, (region, zone)) in enumerate(zip(assignment_uris, plan.regions)):
        vm_name = f"{name_prefix}-{plan.run_id}-{i + 1:02d}".lower().replace("_", "-")
        md = dict(metadata_common)
        md["assignment-uri"] = assign_uri
        print(f"  ▸ creating VM {vm_name} in {zone}...")
        try:
            instance = gcp.create_spot_vm(
                project=cfg.project,
                zone=zone,
                name=vm_name,
                machine_type=spec.machine_type,
                accelerator_type=spec.accelerator_type,
                image_project=cfg.image_project,
                image_family=cfg.image_family,
                startup_script_path=str(startup_script_path),
                metadata=md,
            )
            print(f"    ✓ {vm_name} created (id={instance[0].get('id', '?') if isinstance(instance, list) else instance.get('id', '?')})")
            instances.append(
                {
                    "vm_name": vm_name,
                    "zone": zone,
                    "region": region,
                    "assignment_uri": assign_uri,
                }
            )
        except gcp.GCloudError as e:
            print(f"    ✗ failed to create {vm_name}: {e}", file=sys.stderr)
            instances.append(
                {
                    "vm_name": vm_name,
                    "zone": zone,
                    "region": region,
                    "assignment_uri": assign_uri,
                    "error": str(e),
                }
            )
    return instances
