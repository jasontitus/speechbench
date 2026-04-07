"""speechbench command-line interface (local laptop)."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import click

from . import __version__
from .config import (
    DEFAULT_BUCKET,
    DEFAULT_REGION,
    DEFAULT_ZONE,
    GPUS,
    PROJECT_ID,
    LaunchConfig,
)
from .cost import estimate_run, render_estimate
from .datasets import DATASETS, list_dataset_keys
from .datasets import resolve as resolve_datasets
from .models import MODELS, list_model_keys
from .models import resolve as resolve_models


REPO_ROOT = Path(__file__).resolve().parent.parent


@click.group()
@click.version_option(__version__, prog_name="speechbench")
def cli() -> None:
    """ASR benchmark suite — runs WER/RTFx/GPU-memory benchmarks on GCP spot GPUs."""


# ─── list ──────────────────────────────────────────────────────────────────────


@cli.command(name="list")
@click.option("--what", type=click.Choice(["models", "datasets", "all"]), default="all")
def list_cmd(what: str) -> None:
    """List available models and datasets."""
    if what in ("models", "all"):
        click.echo("─── Models ─────────────────────────────────────────────────────────────────")
        click.echo(f"  {'KEY':30s}  {'FAMILY':14s}  {'BACKEND':16s}  HF id")
        for k in list_model_keys():
            m = MODELS[k]
            click.echo(f"  {k:30s}  {m.family:14s}  {m.backend:16s}  {m.hf_id}")
        click.echo("")
    if what in ("datasets", "all"):
        click.echo("─── Datasets ───────────────────────────────────────────────────────────────")
        click.echo(f"  {'KEY':22s}  {'CAP':>5s}  {'AUTH':>5s}  HF dataset")
        for k in list_dataset_keys():
            d = DATASETS[k]
            auth = "yes" if d.requires_auth else ""
            click.echo(f"  {k:22s}  {d.default_cap:>5d}  {auth:>5s}  {d.hf_dataset}{f' ({d.hf_config})' if d.hf_config else ''}")
        click.echo("")


# ─── estimate ──────────────────────────────────────────────────────────────────


def _resolve_models_arg(models: tuple[str, ...]) -> list[str]:
    if not models:
        # default: every non-API model (DashScope is opt-in)
        return [k for k, v in MODELS.items() if not v.is_api]
    return resolve_models(list(models))


def _resolve_datasets_arg(datasets: tuple[str, ...]) -> list[str]:
    if not datasets:
        # default: ungated long-form + clean baselines
        return [
            "librispeech_clean",
            "librispeech_other",
            "ami_ihm",
            "earnings22",
            "tedlium_longform",
            "voxpopuli_en",
        ]
    return resolve_datasets(list(datasets))


def _build_sample_caps(
    dataset_keys: list[str],
    *,
    quick: bool,
    full: bool,
    sample_cap: Optional[int],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dataset_keys:
        spec = DATASETS[d]
        if full:
            out[d] = spec.full_size or 0
        elif quick:
            out[d] = max(10, min(50, spec.default_cap // 5)) if d != "tedlium_longform" else min(2, spec.default_cap)
        elif sample_cap is not None:
            out[d] = sample_cap
        else:
            out[d] = spec.default_cap
    return out


@cli.command(name="estimate")
@click.option("--models", "-m", multiple=True, help="Model keys or globs (e.g. 'whisper-*'). Default: all non-API.")
@click.option("--datasets", "-d", multiple=True, help="Dataset keys or globs. Default: standard 6.")
@click.option("--gpu", type=click.Choice(list(GPUS.keys())), default="t4")
@click.option("--max-vms", type=int, default=1, help="Number of VMs to fan out across.")
@click.option("--quick", is_flag=True, help="Use small per-dataset caps for a sanity check.")
@click.option("--full", is_flag=True, help="Use the entire test split for each dataset.")
@click.option("--sample-cap", type=int, help="Override per-dataset sample cap (applies to all datasets).")
def estimate_cmd(models, datasets, gpu, max_vms, quick, full, sample_cap):
    """Print a cost / wall-time estimate for a launch without creating any VMs."""
    model_keys = _resolve_models_arg(models)
    dataset_keys = _resolve_datasets_arg(datasets)
    caps = _build_sample_caps(dataset_keys, quick=quick, full=full, sample_cap=sample_cap)

    job_specs = [(m, d, caps[d]) for m in model_keys for d in dataset_keys]
    est = estimate_run(job_specs, gpu_key=gpu, max_vms=max_vms)
    click.echo(render_estimate(est, planned_vms=max_vms, model_keys=model_keys, dataset_keys=dataset_keys))


# ─── launch ────────────────────────────────────────────────────────────────────


@cli.command(name="launch")
@click.option("--run-id", help="Reuse an existing run id (additive). Default: new timestamped id.")
@click.option("--models", "-m", multiple=True)
@click.option("--datasets", "-d", multiple=True)
@click.option("--gpu", type=click.Choice(list(GPUS.keys())), default="t4")
@click.option("--max-vms", type=int, default=1)
@click.option("--multi-region/--single-region", default=True,
              help="Distribute VMs across multiple regions to use per-region quotas (default on).")
@click.option("--quick", is_flag=True)
@click.option("--full", is_flag=True)
@click.option("--sample-cap", type=int)
@click.option("--rerun", is_flag=True, help="Re-run jobs even if results already exist.")
@click.option("--dry-run", is_flag=True, help="Print the plan + cost estimate, do NOT create VMs.")
@click.option("--yes", "-y", is_flag=True, help="Skip the interactive confirmation.")
@click.option("--bucket", default=DEFAULT_BUCKET, show_default=True)
@click.option("--project", default=PROJECT_ID, show_default=True)
@click.option("--region", default=DEFAULT_REGION, show_default=True,
              help="Used when --single-region is set.")
@click.option("--zone", default=DEFAULT_ZONE, show_default=True)
@click.option(
    "--api-key",
    "api_keys",
    multiple=True,
    help="API keys for API-only models, e.g. --api-key DASHSCOPE_API_KEY=sk-...",
)
def launch_cmd(
    run_id,
    models,
    datasets,
    gpu,
    max_vms,
    multi_region,
    quick,
    full,
    sample_cap,
    rerun,
    dry_run,
    yes,
    bucket,
    project,
    region,
    zone,
    api_keys,
):
    """Plan + launch a benchmark run on GCP spot GPU VMs."""
    from . import gcp
    from .orchestrate import plan_launch, upload_assignments, upload_source, launch_vms

    cfg = LaunchConfig(
        project=project,
        bucket=bucket,
        region=region,
        zone=zone,
        gpu=gpu,
        max_vms=max(1, max_vms),
        api_keys=_parse_api_keys(api_keys),
    )

    model_keys = _resolve_models_arg(models)
    dataset_keys = _resolve_datasets_arg(datasets)
    caps = _build_sample_caps(dataset_keys, quick=quick, full=full, sample_cap=sample_cap)

    # Pull DASHSCOPE key from env if any selected model needs it.
    needs_dashscope = any(MODELS[k].needs_dashscope_key for k in model_keys)
    if needs_dashscope and "DASHSCOPE_API_KEY" not in cfg.api_keys:
        env_key = os.environ.get("DASHSCOPE_API_KEY")
        if env_key:
            cfg.api_keys["DASHSCOPE_API_KEY"] = env_key
        else:
            click.secho(
                "  ! one of the selected models needs DASHSCOPE_API_KEY but none was provided. "
                "Pass --api-key DASHSCOPE_API_KEY=... or set the env var.",
                fg="yellow",
            )

    plan = plan_launch(
        cfg=cfg,
        model_keys=model_keys,
        dataset_keys=dataset_keys,
        sample_caps=caps,
        run_id=run_id,
        rerun=rerun,
        multi_region=multi_region,
    )

    click.echo("")
    click.echo(f"Run id:        {plan.run_id}")
    click.echo(f"Bucket:        gs://{plan.bucket}")
    click.echo(f"Models:        {len(model_keys)}  ({', '.join(model_keys[:6])}{'…' if len(model_keys) > 6 else ''})")
    click.echo(f"Datasets:      {len(dataset_keys)}  ({', '.join(dataset_keys)})")
    click.echo(f"Sample caps:   {caps}")
    click.echo(f"Jobs to run:   {len(plan.jobs)}  (skipped already-done: {len(plan.skipped)})")
    click.echo(f"VMs:           {len(plan.bins)}  ({'multi-region' if multi_region else 'single-region'})")
    if plan.regions:
        click.echo(f"VM placements: {plan.regions}")

    if not plan.jobs:
        click.secho("Nothing to do — every requested job already has a result. Use --rerun to force.", fg="green")
        return

    # Cost estimate
    job_specs = [(j.model_key, j.dataset_key, j.sample_cap) for j in plan.jobs]
    est = estimate_run(job_specs, gpu_key=gpu, max_vms=len(plan.bins))
    click.echo("")
    click.echo(render_estimate(est, planned_vms=len(plan.bins), model_keys=model_keys, dataset_keys=dataset_keys))

    if dry_run:
        click.secho("\n--dry-run set; no VMs created.", fg="yellow")
        return

    # Hard check: GPUS_ALL_REGIONS is a project-wide cap that gates EVERY GPU
    # VM regardless of region/type. New GCP projects start at 0 and need a
    # quota raise. We refuse to launch (and waste a tarball upload + bucket
    # create) if it isn't high enough.
    try:
        global_gpu_quota = gcp.project_quota(project, "GPUS_ALL_REGIONS")
    except Exception as e:
        click.secho(f"  ! could not read GPUS_ALL_REGIONS quota: {e}", fg="yellow")
        global_gpu_quota = -1
    needed_vms = len(plan.bins)
    if global_gpu_quota >= 0 and global_gpu_quota < needed_vms:
        click.secho("", fg="red")
        click.secho(
            f"✗ Project '{project}' has GPUS_ALL_REGIONS = {global_gpu_quota}, "
            f"but you're asking for {needed_vms} GPU VM(s).",
            fg="red",
        )
        click.secho(
            "  GCP enforces this project-wide cap on top of the per-region per-type quotas. "
            "Even though open-testimony has 1 spot T4/L4 in each region, GPUS_ALL_REGIONS = 0 "
            "blocks any GPU VM from being created.",
            fg="red",
        )
        click.echo("")
        click.echo("To fix:")
        click.echo("  1. Open the IAM Quotas page:")
        click.echo(f"     https://console.cloud.google.com/iam-admin/quotas?project={project}")
        click.echo("  2. Filter for: 'GPUs (all regions)'  (metric: compute.googleapis.com/gpus_all_regions)")
        click.echo(f"  3. Request a new limit of at least {needed_vms} (suggest 8 for full multi-region parallel).")
        click.echo("  4. While you're there, also raise: NVIDIA_T4_GPUS / PREEMPTIBLE_NVIDIA_T4_GPUS")
        click.echo("     per-region if you want >1 VM in a single region.")
        click.echo("")
        click.echo("Quota raises usually approve in minutes for small numbers (≤8) on long-standing projects;")
        click.echo("brand-new projects sometimes need a billing-history note. After it's granted re-run:")
        click.echo(f"  speechbench launch --run-id {plan.run_id} --gpu {gpu} --max-vms {needed_vms} ...")
        sys.exit(2)

    if not yes:
        click.echo("")
        click.confirm("Proceed and create the spot VM(s)?", abort=True, default=False)

    click.echo("\n▶ ensuring GCS bucket exists")
    gcp.ensure_bucket(plan.bucket, project=project, region=region)

    click.echo("▶ packaging source and uploading to GCS")
    upload_source(plan, REPO_ROOT)

    click.echo("▶ writing per-VM assignments")
    assignment_uris = upload_assignments(plan)

    click.echo("▶ creating spot VMs")
    startup_path = REPO_ROOT / "speechbench" / "startup.sh"
    instances = launch_vms(
        plan,
        cfg,
        assignment_uris,
        rerun=rerun,
        startup_script_path=startup_path,
    )

    click.echo("\nLaunched:")
    for inst in instances:
        click.echo(f"  • {inst['vm_name']} in {inst['zone']}  (assignment={inst['assignment_uri']})")
    click.echo("")
    click.echo("Watch progress:")
    click.echo(f"  speechbench status --run-id {plan.run_id}")
    click.echo("Then once VMs finish:")
    click.echo(f"  speechbench report --run-id {plan.run_id}")


# ─── status ────────────────────────────────────────────────────────────────────


@cli.command(name="run")
@click.option("--run-id", default=None, help="Run id (default: timestamped). Local results land in ./results/<run-id>/")
@click.option("--models", "-m", multiple=True)
@click.option("--datasets", "-d", multiple=True)
@click.option("--quick", is_flag=True)
@click.option("--full", is_flag=True)
@click.option("--sample-cap", type=int)
@click.option("--rerun", is_flag=True, help="Re-run jobs even if a result already exists.")
@click.option("--out-dir", default=None, help="Local results dir (default: ./results/<run-id>/)")
@click.option(
    "--api-key",
    "api_keys",
    multiple=True,
    help="API keys for API-only models, e.g. --api-key DASHSCOPE_API_KEY=sk-...",
)
def run_cmd(run_id, models, datasets, quick, full, sample_cap, rerun, out_dir, api_keys):
    """Run benchmarks locally (no GCP). Works on Apple Silicon and Linux."""
    import datetime as _dt
    import json
    import os as _os
    import secrets
    from pathlib import Path as _Path

    from .datasets import DATASETS
    from .models import MODELS, is_mac_arm
    from .orchestrate import job_id_for

    rid = run_id or _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S") + "-local-" + secrets.token_hex(2)
    out_path = _Path(out_dir) if out_dir else REPO_ROOT / "results" / rid
    out_path.mkdir(parents=True, exist_ok=True)
    results_dir = out_path / "raw"
    results_dir.mkdir(exist_ok=True)

    model_keys = _resolve_models_arg(models)
    dataset_keys = _resolve_datasets_arg(datasets)
    caps = _build_sample_caps(dataset_keys, quick=quick, full=full, sample_cap=sample_cap)

    # Set API keys into env so the DashScope wrapper can read them
    for kv in api_keys:
        if "=" in kv:
            k, v = kv.split("=", 1)
            _os.environ[k.strip()] = v.strip()

    # Build job list, skip already-done unless --rerun
    jobs = []
    for m in model_keys:
        for d in dataset_keys:
            cap = caps[d]
            jid = job_id_for(m, d, cap)
            result_path = results_dir / f"{jid}.json"
            if result_path.exists() and not rerun:
                click.echo(f"⤿ skip (already done): {m} × {d}@{cap}")
                continue
            jobs.append({"job_id": jid, "model_key": m, "dataset_key": d, "sample_cap": cap})

    click.echo("")
    click.echo(f"Run id:    {rid}")
    click.echo(f"Out dir:   {out_path}")
    click.echo(f"Platform:  {'darwin/arm64 (Mac, MLX)' if is_mac_arm() else 'linux/cuda or cpu'}")
    click.echo(f"Models:    {len(model_keys)}")
    click.echo(f"Datasets:  {len(dataset_keys)}  ({', '.join(dataset_keys)})")
    click.echo(f"Jobs:      {len(jobs)}")
    if not jobs:
        click.echo("Nothing to run.")
        return

    # Save the job list locally for the report command
    (out_path / "jobs.json").write_text(json.dumps({"run_id": rid, "jobs": jobs}, indent=2))

    # Drive the runner directly. We import lazily so the local CLI starts fast
    # for `list` / `estimate` even if torch / nemo aren't installed.
    from .eval import GPUMemorySampler
    from .runner import run_job_local

    sampler = GPUMemorySampler()
    sampler.start()
    completed = 0
    failed = 0
    skipped = 0

    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        grouped.setdefault(job["model_key"], []).append(job)

    for model_key, group in grouped.items():
        try:
            from .models import MODELS, make_model

            spec = MODELS[model_key]
            click.echo(f"\n▶▶ loading {spec.hf_id} ({spec.backend})")
            sampler.reset_peak()
            pre_load_baseline = sampler.baseline_mb
            import time as _time

            model = make_model(spec)
            t_load_start = _time.time()
            try:
                model.load()
            except Exception as e:
                click.secho(f"   ✗ load failed: {e}", fg="red")
                failed += len(group)
                for job in group:
                    err = {
                        "job_id": job["job_id"],
                        "model_key": model_key,
                        "dataset_key": job["dataset_key"],
                        "sample_cap": job["sample_cap"],
                        "failed": True,
                        "stage": "load",
                        "error": str(e)[:1000],
                    }
                    (results_dir / f"{job['job_id']}.failed.json").write_text(
                        json.dumps(err, indent=2)
                    )
                continue
            load_seconds = _time.time() - t_load_start
            load_mb = max(0.0, sampler.peak_mb - pre_load_baseline)
            click.echo(f"   loaded in {load_seconds:.1f}s (+{load_mb:.0f} MB)")

            for job in group:
                try:
                    res = run_job_local(
                        model,
                        load_seconds,
                        load_mb,
                        job["dataset_key"],
                        int(job["sample_cap"]),
                        job["job_id"],
                        sampler,
                    )
                    (results_dir / f"{job['job_id']}.json").write_text(
                        json.dumps(res.to_dict(), indent=2, default=str)
                    )
                    completed += 1
                except Exception as e:
                    click.secho(f"   ✗ {job['dataset_key']}: {e}", fg="red")
                    failed += 1
                    err = {
                        "job_id": job["job_id"],
                        "model_key": model_key,
                        "dataset_key": job["dataset_key"],
                        "sample_cap": job["sample_cap"],
                        "failed": True,
                        "error": str(e)[:1000],
                    }
                    (results_dir / f"{job['job_id']}.failed.json").write_text(
                        json.dumps(err, indent=2)
                    )

            try:
                model.unload()
            except Exception:
                pass
        except KeyError:
            click.secho(f"   ! unknown model {model_key}", fg="yellow")
            failed += len(group)

    sampler.stop()
    click.echo(f"\nDone. completed={completed} failed={failed} skipped={skipped} total={len(jobs)}")
    click.echo("")
    click.echo(f"Render report: speechbench report --run-id {rid} --bucket {out_path}")
    click.echo(f"Or open: {out_path}/")


@cli.command(name="status")
@click.option("--run-id", required=True)
@click.option("--bucket", default=DEFAULT_BUCKET)
@click.option("--project", default=PROJECT_ID)
def status_cmd(run_id, bucket, project):
    """Show how many jobs are done / pending for a run."""
    from . import gcp

    bucket_uri = bucket if bucket.startswith("gs://") else f"gs://{bucket}"

    jobs_payload = gcp.gs_download_string(f"{bucket_uri}/runs/{run_id}/jobs.json")
    if not jobs_payload:
        click.secho(f"No jobs.json found at {bucket_uri}/runs/{run_id}/", fg="red")
        sys.exit(1)
    jobs = json.loads(jobs_payload).get("jobs", [])
    results = gcp.gs_list(f"{bucket_uri}/runs/{run_id}/results/")
    done_ids = {Path(r).name.removesuffix(".json").removesuffix(".failed") for r in results if r.endswith(".json")}

    todo = [j for j in jobs if j["job_id"] not in done_ids]
    done = [j for j in jobs if j["job_id"] in done_ids]
    failed = [r for r in results if r.endswith(".failed.json")]
    click.echo(f"run {run_id}: total={len(jobs)} done={len(done)} pending={len(todo)} failed={len(failed)}")
    click.echo("")
    if todo:
        click.echo("pending:")
        for j in todo:
            click.echo(f"  - {j['model_key']} × {j['dataset_key']}@{j['sample_cap']}  ({j['job_id']})")
    instances = gcp.list_instances(project)
    live = [i for i in instances if i.get("name", "").startswith("speechbench-") and run_id in i.get("name", "")]
    if live:
        click.echo("\nlive VMs:")
        for i in live:
            click.echo(f"  - {i['name']}  zone={i.get('zone', '?').rsplit('/', 1)[-1]}  status={i.get('status', '?')}")


# ─── report ────────────────────────────────────────────────────────────────────


@cli.command(name="report")
@click.option("--run-id", required=True)
@click.option("--bucket", default=DEFAULT_BUCKET)
@click.option("--out", "out_dir", default=None, help="Local output directory (default: results/<run-id>/)")
def report_cmd(run_id, bucket, out_dir):
    """Pull all result JSONs from GCS and build a comparative report."""
    from .report import render_console_table, write_report

    out_path = Path(out_dir) if out_dir else REPO_ROOT / "results" / run_id
    info = write_report(bucket, run_id, out_path)
    click.echo(f"▶ pulled {info['summary']['num_results']} result file(s)")
    click.echo(f"▶ wrote {out_path}/report.md")
    click.echo(f"▶ wrote {out_path}/report.csv")
    click.echo(f"▶ wrote {out_path}/summary.json")
    click.echo("")
    click.echo(render_console_table(info["results"]))


# ─── cancel ────────────────────────────────────────────────────────────────────


@cli.command(name="cancel")
@click.option("--run-id", required=True)
@click.option("--project", default=PROJECT_ID)
@click.option("--yes", "-y", is_flag=True)
def cancel_cmd(run_id, project, yes):
    """Delete any live VMs for the given run id."""
    from . import gcp

    instances = gcp.list_instances(project)
    targets = [i for i in instances if "speechbench-" in i.get("name", "") and run_id in i.get("name", "")]
    if not targets:
        click.echo("No matching VMs.")
        return
    click.echo(f"Will delete {len(targets)} VM(s):")
    for i in targets:
        click.echo(f"  - {i['name']} (zone={i.get('zone', '').rsplit('/', 1)[-1]})")
    if not yes:
        click.confirm("Proceed?", abort=True, default=False)
    for i in targets:
        zone = i["zone"].rsplit("/", 1)[-1]
        try:
            gcp.delete_instance(project, zone, i["name"])
            click.echo(f"  ✓ deleted {i['name']}")
        except Exception as e:
            click.echo(f"  ✗ {i['name']}: {e}")


# ─── helpers ───────────────────────────────────────────────────────────────────


def _parse_api_keys(pairs: tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise click.BadParameter(f"--api-key must be KEY=VALUE, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = v.strip()
    return out


if __name__ == "__main__":
    cli()
