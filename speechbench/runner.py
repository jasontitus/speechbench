"""VM-side benchmark runner.

Reads an `assignment.json` from GCS, runs each (model, dataset, sample_cap)
job, writes a result JSON for each, and uploads it back to GCS. Skips jobs
whose result file already exists in GCS unless `--rerun` is set.

Invocation (from startup.sh):

    python -m speechbench.runner \
        --assignment-uri gs://bucket/runs/<run_id>/assignments/<vm>.json \
        --bucket gs://bucket \
        --run-id <run_id> \
        [--rerun]
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
from typing import Optional

import signal
import threading
from contextlib import contextmanager

from .datasets import DATASETS, load
from .eval import (
    ClipResult,
    GPUMemorySampler,
    JobResult,
    compute_cer,
    compute_wer,
    mean,
    normalize_text,
    per_clip_cer,
    per_clip_wer,
    percentile,
)
from .models import MODELS, ModelSpec, make_model


# ─── Wallclock timeout (so a stuck model can't lock up the VM forever) ────────


class TimeoutError_(Exception):
    pass


@contextmanager
def time_limit(seconds: int):
    """SIGALRM-based wallclock timeout. Only safe in the main thread."""

    def _handler(signum, frame):
        raise TimeoutError_(f"timed out after {seconds}s")

    if threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
    else:
        # Best effort if we're not on the main thread.
        yield


# ─── GCS helpers (uses gsutil so we don't depend on the python SDK on the VM) ──


def gsutil_cp(local: str, remote: str) -> None:
    subprocess.run(["gsutil", "-q", "cp", local, remote], check=True)


def gsutil_cat(remote: str) -> Optional[bytes]:
    try:
        out = subprocess.run(
            ["gsutil", "-q", "cat", remote], check=True, capture_output=True
        )
        return out.stdout
    except subprocess.CalledProcessError:
        return None


def gsutil_exists(remote: str) -> bool:
    out = subprocess.run(
        ["gsutil", "-q", "stat", remote], capture_output=True
    )
    return out.returncode == 0


def upload_json(obj: dict, remote: str) -> None:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(obj, f, indent=2, default=str)
        local = f.name
    try:
        gsutil_cp(local, remote)
    finally:
        try:
            os.unlink(local)
        except OSError:
            pass


# ─── Single job runner ─────────────────────────────────────────────────────────


# Public alias used by the local CLI (`speechbench run`).
def run_job_local(*args, **kwargs):
    return run_job(*args, **kwargs)


def run_job(
    model,
    model_load_seconds: float,
    model_load_mem_mb: float,
    dataset_key: str,
    sample_cap: int,
    job_id: str,
    sampler: GPUMemorySampler,
) -> JobResult:
    """Run a single (model, dataset) job using an *already-loaded* model.

    The driver loop is responsible for loading and unloading models so we can
    amortize the load cost across all the (model × N datasets) jobs that share
    a model_key.
    """
    spec: ModelSpec = model.spec
    dataset_spec = DATASETS[dataset_key]

    result = JobResult(
        job_id=job_id,
        model_key=spec.key,
        model_id=spec.hf_id,
        backend=spec.backend,
        family=spec.family,
        dataset_key=dataset_key,
        sample_cap=sample_cap,
        gpu_name=sampler.info.name,
        gpu_compute_capability=sampler.info.compute_capability,
        gpu_total_mb=sampler.info.total_mb,
        model_load_seconds=model_load_seconds,
        model_load_mem_mb=model_load_mem_mb,
    )
    result.started_at = time.time()
    print(f"\n══ {spec.key} × {dataset_key}@{sample_cap} ══", flush=True)

    # Reset the GPU sampler so the per-job peak measures only inference, not
    # the (already-amortized) model load.
    sampler.reset_peak()
    pre_inference_baseline = sampler.baseline_mb

    # Stream the dataset
    refs: list[str] = []
    hyps: list[str] = []
    latencies: list[float] = []
    rtfx_values: list[float] = []
    failed = 0

    sampler.reset_peak()
    inference_start = time.time()

    for clip_id, audio, ref_raw, audio_seconds in load(dataset_spec, sample_cap):
        try:
            t0 = time.time()
            # Per-clip timeout: 5 minutes is generous for any reasonable model.
            with time_limit(300):
                hyp_raw = model.transcribe(audio, sample_rate=16000)
            t1 = time.time()
            latency_ms = (t1 - t0) * 1000.0
            ref_norm = normalize_text(ref_raw)
            hyp_norm = normalize_text(hyp_raw)

            wer = per_clip_wer(ref_raw, hyp_raw)
            cer = per_clip_cer(ref_raw, hyp_raw)
            rtfx = (audio_seconds / max(1e-6, (t1 - t0))) if audio_seconds > 0 else 0.0

            refs.append(ref_raw)
            hyps.append(hyp_raw)
            latencies.append(latency_ms)
            rtfx_values.append(rtfx)
            result.audio_duration_s_total += audio_seconds

            result.clips.append(
                ClipResult(
                    clip_id=clip_id,
                    audio_seconds=audio_seconds,
                    reference_raw=ref_raw,
                    reference_norm=ref_norm,
                    hypothesis_raw=hyp_raw,
                    hypothesis_norm=hyp_norm,
                    latency_ms=latency_ms,
                    wer=wer,
                    cer=cer,
                )
            )
            if (len(result.clips) % 25) == 0:
                print(
                    f"    [{len(result.clips):4d}] "
                    f"avg_wer={compute_wer(refs, hyps):.3f}  "
                    f"rtfx_mean={mean(rtfx_values):.1f}",
                    flush=True,
                )
        except Exception as e:
            failed += 1
            traceback.print_exc()
            result.clips.append(
                ClipResult(
                    clip_id=clip_id,
                    audio_seconds=audio_seconds,
                    reference_raw=ref_raw,
                    reference_norm=normalize_text(ref_raw),
                    hypothesis_raw="",
                    hypothesis_norm="",
                    latency_ms=0.0,
                    wer=1.0,
                    cer=1.0,
                    failed=True,
                    error=str(e)[:500],
                )
            )

    result.num_clips = len(result.clips)
    result.failed_clips = failed
    result.gpu_peak_mem_mb = max(0.0, sampler.peak_mb - pre_inference_baseline)
    result.wall_time_s = time.time() - result.started_at
    result.finished_at = time.time()

    if refs:
        result.wer = compute_wer(refs, hyps)
        result.cer = compute_cer(refs, hyps)
    if latencies:
        result.latency_ms_mean = mean(latencies)
        result.latency_ms_p50 = percentile(latencies, 50)
        result.latency_ms_p90 = percentile(latencies, 90)
    if rtfx_values:
        result.rtfx_mean = mean(rtfx_values)
        result.rtfx_p50 = percentile(rtfx_values, 50)
        result.rtfx_p10 = percentile(rtfx_values, 10)

    print(
        f"  ✓ done in {result.wall_time_s:.1f}s — "
        f"WER={result.wer:.3f} CER={result.cer:.3f} "
        f"RTFx={result.rtfx_mean:.1f} "
        f"GPU peak={result.gpu_peak_mem_mb:.0f} MB",
        flush=True,
    )

    return result


# ─── Driver ────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--assignment-uri", required=True)
    p.add_argument("--bucket", required=True, help="gs://bucket/path")
    p.add_argument("--run-id", required=True)
    p.add_argument("--rerun", action="store_true")
    p.add_argument("--vm-name", default=socket.gethostname())
    args = p.parse_args()

    bucket = args.bucket.rstrip("/")
    results_prefix = f"{bucket}/runs/{args.run_id}/results"

    # Pull assignment
    raw = gsutil_cat(args.assignment_uri)
    if raw is None:
        print(f"failed to read assignment from {args.assignment_uri}", file=sys.stderr)
        return 2
    assignment = json.loads(raw.decode())
    jobs = assignment.get("jobs", [])
    print(f"VM {args.vm_name} has {len(jobs)} job(s) assigned.", flush=True)

    sampler = GPUMemorySampler()
    sampler.start()

    # Group jobs by model_key so each model is loaded once and reused for all
    # of its (model × dataset) jobs. Inside a model group, run datasets in the
    # order given.
    grouped: dict[str, list[dict]] = {}
    for job in jobs:
        grouped.setdefault(job["model_key"], []).append(job)

    completed = 0
    failed = 0
    skipped = 0

    for model_key, group in grouped.items():
        # Filter out already-done jobs first; if everything is done for this
        # model we never load the weights.
        pending: list[dict] = []
        for job in group:
            remote_result = f"{results_prefix}/{job['job_id']}.json"
            if not args.rerun and gsutil_exists(remote_result):
                print(
                    f"⤿ skip (already done): {model_key} × {job['dataset_key']}@{job['sample_cap']}",
                    flush=True,
                )
                skipped += 1
            else:
                pending.append(job)

        if not pending:
            continue

        if model_key not in MODELS:
            print(f"✗ unknown model: {model_key}", flush=True)
            failed += len(pending)
            continue

        spec = MODELS[model_key]
        print(f"\n▶▶ loading {spec.hf_id} ({spec.backend})", flush=True)
        sampler.reset_peak()
        pre_load_baseline = sampler.baseline_mb
        model = make_model(spec)
        t_load_start = time.time()
        # Hard cap on model load (10 min) so a stuck model can't lock up the VM.
        try:
            with time_limit(600):
                model.load()
        except Exception as e:
            traceback.print_exc()
            failed += len(pending)
            err = {
                "model_key": model_key,
                "model_id": spec.hf_id,
                "failed": True,
                "stage": "load",
                "error": str(e)[:1000],
            }
            for job in pending:
                try:
                    upload_json(
                        {**err, "job_id": job["job_id"], "dataset_key": job["dataset_key"], "sample_cap": job["sample_cap"]},
                        f"{results_prefix}/{job['job_id']}.failed.json",
                    )
                except Exception:
                    pass
            continue
        load_seconds = time.time() - t_load_start
        load_mb = max(0.0, sampler.peak_mb - pre_load_baseline)
        print(
            f"   loaded in {load_seconds:.1f}s (+{load_mb:.0f} MB GPU)",
            flush=True,
        )

        for job in pending:
            try:
                res = run_job(
                    model,
                    load_seconds,
                    load_mb,
                    job["dataset_key"],
                    int(job["sample_cap"]),
                    job["job_id"],
                    sampler,
                )
                upload_json(res.to_dict(), f"{results_prefix}/{job['job_id']}.json")
                if res.num_clips == 0 and res.failed_clips != 0:
                    failed += 1
                else:
                    completed += 1
            except Exception as e:
                traceback.print_exc()
                failed += 1
                try:
                    upload_json(
                        {
                            "job_id": job["job_id"],
                            "model_key": model_key,
                            "dataset_key": job["dataset_key"],
                            "sample_cap": job["sample_cap"],
                            "failed": True,
                            "error": str(e)[:1000],
                        },
                        f"{results_prefix}/{job['job_id']}.failed.json",
                    )
                except Exception:
                    pass

        # Free model memory before loading the next one
        try:
            model.unload()
        except Exception:
            pass
        del model
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    sampler.stop()
    summary = {
        "vm_name": args.vm_name,
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "total": len(jobs),
        "finished_at": time.time(),
    }
    try:
        upload_json(summary, f"{bucket}/runs/{args.run_id}/logs/{args.vm_name}.summary.json")
    except Exception:
        pass

    print(
        f"\nDone. completed={completed} failed={failed} skipped={skipped} total={len(jobs)}",
        flush=True,
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
