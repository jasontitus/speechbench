"""Local report generator.

Pulls every results/*.json from gs://<bucket>/runs/<run_id>/results and
emits:
  results/<run_id>/report.md     human-readable Markdown
  results/<run_id>/report.csv    flat CSV
  results/<run_id>/summary.json  machine-readable summary

The Markdown report has one table per dataset, sorted by the language's
primary metric (WER for English, CER for non-English).

Aggregate WER/CER are *recomputed from per-clip raw fields* after every
fetch so the report always reflects the current normalizer, even if the
raw JSON in GCS was written by an older version of the suite.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from . import gcp
from .config import DEFAULT_BUCKET
from .datasets import DATASETS
from .eval import compute_cer, compute_wer, normalize_text, per_clip_cer, per_clip_wer


def _language_for_dataset(dataset_key: str) -> str:
    spec = DATASETS.get(dataset_key)
    if spec is None:
        return "english"
    return getattr(spec, "language", "english") or "english"


def _recompute_in_place(j: dict) -> None:
    """Re-derive aggregate WER/CER from per-clip raw fields using the current
    normalizer. Mutates `j` in place. No-op if there are no clips."""
    clips = j.get("clips") or []
    if not clips:
        return
    lang = _language_for_dataset(j.get("dataset_key", ""))
    refs: list[str] = []
    hyps: list[str] = []
    for c in clips:
        ref = c.get("reference_raw", "") or ""
        if c.get("failed"):
            refs.append(ref)
            hyps.append("")
            c["wer"] = 1.0
            c["cer"] = 1.0
            c["reference_norm"] = normalize_text(ref, language=lang)
            c["hypothesis_norm"] = ""
            continue
        hyp = c.get("hypothesis_raw", "") or ""
        refs.append(ref)
        hyps.append(hyp)
        c["reference_norm"] = normalize_text(ref, language=lang)
        c["hypothesis_norm"] = normalize_text(hyp, language=lang)
        c["wer"] = per_clip_wer(ref, hyp, language=lang)
        c["cer"] = per_clip_cer(ref, hyp, language=lang)
    j["wer"] = compute_wer(refs, hyps, language=lang)
    j["cer"] = compute_cer(refs, hyps, language=lang)
    j["language"] = lang


def fetch_results(bucket: str, run_id: str, local_dir: Path) -> list[dict]:
    """Pull result JSONs.

    `bucket` may be either a GCS bucket name (or `gs://...` URI) OR a local
    filesystem path. The local-path branch is used by `speechbench run --local`.

    Every result file is recomputed in place against the current normalizer
    (so a normalizer fix automatically reflects in subsequent reports without
    needing to re-run inference).
    """
    bucket_str = str(bucket)

    # Local-path mode: a directory we can read directly.
    bucket_path = Path(bucket_str)
    if bucket_str.startswith("/") or bucket_path.is_dir():
        # Two layouts we accept:
        #   <bucket>/raw/<job_id>.json          — what `speechbench run` writes
        #   <bucket>/runs/<run_id>/results/*    — same as the GCS layout, mirrored
        candidates = [
            bucket_path / "raw",
            bucket_path / "runs" / run_id / "results",
            bucket_path,  # in case the user pointed at the results dir directly
        ]
        for cand in candidates:
            if cand.is_dir() and any(cand.glob("*.json")):
                out: list[dict] = []
                for f in sorted(cand.glob("*.json")):
                    if f.name == "jobs.json":
                        continue
                    try:
                        with open(f) as fh:
                            j = json.load(fh)
                    except Exception as e:
                        print(f"  ! failed to read {f}: {e}")
                        continue
                    _recompute_in_place(j)
                    # Persist the recomputed values so downstream tools see them.
                    try:
                        f.write_text(json.dumps(j, indent=2, ensure_ascii=False))
                    except Exception:
                        pass
                    out.append(j)
                return out
        return []

    # GCS mode
    bucket_uri = bucket_str if bucket_str.startswith("gs://") else f"gs://{bucket_str}"
    prefix = f"{bucket_uri}/runs/{run_id}/results/"
    listing = gcp.gs_list(prefix)
    if not listing:
        return []

    local_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for uri in listing:
        if not uri.endswith(".json"):
            continue
        local_path = local_dir / Path(uri).name
        try:
            gcp.gs_download(uri, str(local_path))
            with open(local_path) as f:
                j = json.load(f)
        except Exception as e:
            print(f"  ! failed to fetch {uri}: {e}")
            continue
        _recompute_in_place(j)
        try:
            local_path.write_text(json.dumps(j, indent=2, ensure_ascii=False))
        except Exception:
            pass
        out.append(j)
    return out


def _is_english_dataset(results_for_ds: list[dict]) -> bool:
    """Look at any clip in the dataset's results to figure out the language.

    Result JSONs written by `runner.py` after the normalizer fix store a
    `language` field at the top level. Older files default to English (the
    safe historical assumption).
    """
    for r in results_for_ds:
        lang = r.get("language") or ""
        if lang:
            return lang == "english"
    return True


def render_markdown(results: list[dict], run_id: str) -> str:
    # group by dataset
    by_dataset: dict[str, list[dict]] = {}
    for r in results:
        if r.get("failed") and not r.get("dataset_key"):
            continue
        ds = r.get("dataset_key", "unknown")
        by_dataset.setdefault(ds, []).append(r)

    lines: list[str] = []
    lines.append(f"# speechbench report — run `{run_id}`")
    lines.append("")
    lines.append(f"Pulled {len(results)} result file(s).")
    lines.append("")
    if not by_dataset:
        lines.append("_No completed jobs yet._")
        return "\n".join(lines)

    for ds in sorted(by_dataset):
        # For non-English datasets, sort by CER and put CER before WER —
        # CER is the more informative metric for morphologically rich
        # languages where the model is usually nearly right but loses on
        # word-boundary inflection matches.
        english = _is_english_dataset(by_dataset[ds])
        primary_key = "wer" if english else "cer"
        rows = sorted(by_dataset[ds], key=lambda r: r.get(primary_key, 1.0))
        lines.append(f"## {ds}")
        lines.append("")
        if english:
            lines.append("| Model | Backend | n | WER | CER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |")
        else:
            lines.append("| Model | Backend | n | CER | WER | RTFx (mean) | RTFx (p50) | Latency mean (ms) | GPU peak (MB) | Wall (s) |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            cells = {
                "model": r.get("model_key", "?"),
                "backend": r.get("backend", "?"),
                "n": r.get("num_clips", 0),
                "wer": f"{r.get('wer', 0):.3f}",
                "cer": f"{r.get('cer', 0):.3f}",
                "rtfx": f"{r.get('rtfx_mean', 0):.1f}",
                "rtfxp50": f"{r.get('rtfx_p50', 0):.1f}",
                "lat": f"{r.get('latency_ms_mean', 0):.0f}",
                "gpu": f"{r.get('gpu_peak_mem_mb', 0):.0f}",
                "wall": f"{r.get('wall_time_s', 0):.0f}",
            }
            if english:
                fmt = (
                    "| {model} | {backend} | {n} | {wer} | {cer} | {rtfx} | {rtfxp50} | "
                    "{lat} | {gpu} | {wall} |"
                )
            else:
                fmt = (
                    "| {model} | {backend} | {n} | {cer} | {wer} | {rtfx} | {rtfxp50} | "
                    "{lat} | {gpu} | {wall} |"
                )
            lines.append(fmt.format(**cells))
        lines.append("")
    return "\n".join(lines)


def render_csv_rows(results: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in results:
        if r.get("failed") and not r.get("model_key"):
            continue
        out.append(
            {
                "model_key": r.get("model_key", ""),
                "model_id": r.get("model_id", ""),
                "backend": r.get("backend", ""),
                "family": r.get("family", ""),
                "dataset_key": r.get("dataset_key", ""),
                "sample_cap": r.get("sample_cap", 0),
                "num_clips": r.get("num_clips", 0),
                "failed_clips": r.get("failed_clips", 0),
                "wer": r.get("wer", ""),
                "cer": r.get("cer", ""),
                "rtfx_mean": r.get("rtfx_mean", ""),
                "rtfx_p50": r.get("rtfx_p50", ""),
                "rtfx_p10": r.get("rtfx_p10", ""),
                "latency_ms_mean": r.get("latency_ms_mean", ""),
                "latency_ms_p50": r.get("latency_ms_p50", ""),
                "latency_ms_p90": r.get("latency_ms_p90", ""),
                "gpu_peak_mem_mb": r.get("gpu_peak_mem_mb", ""),
                "model_load_mem_mb": r.get("model_load_mem_mb", ""),
                "wall_time_s": r.get("wall_time_s", ""),
                "audio_duration_s_total": r.get("audio_duration_s_total", ""),
                "gpu_name": r.get("gpu_name", ""),
            }
        )
    return out


def write_report(
    bucket: str,
    run_id: str,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    results = fetch_results(bucket, run_id, raw_dir)

    md = render_markdown(results, run_id)
    (out_dir / "report.md").write_text(md)

    rows = render_csv_rows(results)
    csv_path = out_dir / "report.csv"
    if rows:
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    else:
        csv_path.write_text("")

    summary = {
        "run_id": run_id,
        "bucket": bucket,
        "num_results": len(results),
        "datasets": sorted({r.get("dataset_key", "") for r in results if r.get("dataset_key")}),
        "models": sorted({r.get("model_key", "") for r in results if r.get("model_key")}),
        "best_per_dataset": _best_per_dataset(results),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"results": results, "summary": summary, "out_dir": str(out_dir)}


def _best_per_dataset(results: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    by_ds: dict[str, list[dict]] = {}
    for r in results:
        if r.get("failed"):
            continue
        ds = r.get("dataset_key")
        if not ds:
            continue
        by_ds.setdefault(ds, []).append(r)
    for ds, rows in by_ds.items():
        winner = min(rows, key=lambda r: r.get("wer", 1.0))
        fastest = max(rows, key=lambda r: r.get("rtfx_mean", 0.0))
        smallest = min(rows, key=lambda r: r.get("gpu_peak_mem_mb", 1e9))
        out[ds] = {
            "best_wer": {"model": winner.get("model_key"), "wer": winner.get("wer")},
            "fastest_rtfx": {"model": fastest.get("model_key"), "rtfx_mean": fastest.get("rtfx_mean")},
            "smallest_gpu": {
                "model": smallest.get("model_key"),
                "gpu_peak_mem_mb": smallest.get("gpu_peak_mem_mb"),
            },
        }
    return out


def render_console_table(results: list[dict]) -> str:
    """Pretty-print one block per dataset for the terminal."""
    from tabulate import tabulate

    out: list[str] = []
    by_ds: dict[str, list[dict]] = {}
    for r in results:
        if r.get("failed"):
            continue
        ds = r.get("dataset_key")
        if not ds:
            continue
        by_ds.setdefault(ds, []).append(r)
    for ds in sorted(by_ds):
        english = _is_english_dataset(by_ds[ds])
        primary_key = "wer" if english else "cer"
        rows = sorted(by_ds[ds], key=lambda r: r.get(primary_key, 1.0))
        if english:
            headers = ["model", "backend", "n", "WER", "CER", "RTFx", "lat ms", "GPU MB"]
        else:
            # Non-English: CER first, since one wrong inflection letter is
            # 1 word error but only 1 char error.
            headers = ["model", "backend", "n", "CER", "WER", "RTFx", "lat ms", "GPU MB"]
        table = []
        for r in rows:
            wer_str = f"{r.get('wer', 0)*100:.2f}%"
            cer_str = f"{r.get('cer', 0)*100:.2f}%"
            row = [
                r.get("model_key"),
                r.get("backend"),
                r.get("num_clips"),
            ]
            if english:
                row += [wer_str, cer_str]
            else:
                row += [cer_str, wer_str]
            row += [
                f"{r.get('rtfx_mean', 0):.1f}",
                f"{r.get('latency_ms_mean', 0):.0f}",
                f"{r.get('gpu_peak_mem_mb', 0):.0f}",
            ]
            table.append(row)
        out.append("")
        out.append(f"=== {ds} ===")
        out.append(tabulate(table, headers=headers, tablefmt="simple"))
    return "\n".join(out)
