# speechbench

A comparative benchmark suite for speech-to-text models on **GCP spot GPUs**.
Spins up one or more spot T4 / L4 instances in the `open-testimony` project,
runs every selected model against every selected dataset, and produces a
single comparative report covering **WER**, **CER**, **RTFx** (real-time
factor), **GPU peak memory**, and **per-clip latency**.

## What it benchmarks

### Models (26 total)

- **Whisper** — OpenAI `tiny.en` / `base.en` / `small.en` / `medium.en`,
  `large-v2`, `large-v3`, `large-v3-turbo`, plus `distil-whisper/distil-large-v3`,
  via both HF Transformers and faster-whisper (CTranslate2).
- **Parakeet (NVIDIA NeMo)** — `tdt-0.6b-v2`, `tdt-0.6b-v3`, `tdt-1.1b`,
  `rnnt-0.6b`, `rnnt-1.1b`, `ctc-0.6b`, `ctc-1.1b`, `tdt_ctc-110m`.
- **Qwen3-ASR** — 0.6B and 1.7B (HuggingFace Transformers).
- **Qwen3.5-Omni** — Plus / Flash / Light (DashScope API).
- **Gemma 4 multimodal** — `google/gemma-4-E4B-it` and `gemma-4-E2B-it`,
  audio-prompted via `AutoModelForMultimodalLM`.

`speechbench list --what models` shows the full registry.

### Datasets (8 total)

`librispeech_clean`, `librispeech_other`, `ami_ihm`, `earnings22`,
`tedlium_longform`, `voxpopuli_en`, plus gated `gigaspeech_l` and
`spgispeech` (need an HF token).

`speechbench list --what datasets` shows the full registry.

## How it works

```
laptop                                       GCP open-testimony project
─────────                                    ─────────────────────────────
speechbench launch                           ┌───────────────────────────┐
  • build job matrix                         │ gs://open-testimony-      │
  • diff against existing results in GCS     │     speechbench/          │
  • estimate wall time + spot $ + confirm    │   src/<run>.tar.gz        │
  • tar src → GCS                            │   runs/<run>/jobs.json    │
  • create N spot T4/L4 VMs across regions ──┤   runs/<run>/results/*    │
                                             │   runs/<run>/logs/*       │
speechbench status                           └───────────────┬───────────┘
  • list pending / done / failed             ┌───────────────┴───────────┐
                                             │ N spot GPU VMs            │
speechbench report                           │ (us-central1, us-east1,…) │
  • pull all results                         │  • DLVM PyTorch image     │
  • build report.md / .csv / summary.json    │  • install requirements   │
  • print per-dataset table                  │  • run python -m          │
                                             │      speechbench.runner   │
                                             │  • upload result/<id>.json│
                                             │  • shutdown -h            │
                                             └───────────────────────────┘
```

Key properties:

- **Additive runs.** Re-launching with the same `--run-id` only runs the
  (model × dataset) combinations whose result JSON does NOT already exist in
  GCS. Adding a new dataset and re-running fills in only the new
  combinations.
- **Multi-region parallel.** With `--max-vms N` the orchestrator distributes
  VMs across `us-central1`, `us-east1`, `us-west1`, `us-east4`, `us-west4`,
  `europe-west1`, `europe-west4`, `asia-east1`, `asia-southeast1` — each
  region has its own 1 × T4 + 1 × L4 spot quota in `open-testimony`, so
  this requires no quota raise.
- **Cost confirmation.** `speechbench launch` always prints an estimate
  (jobs, VMs, wall hours per VM, total $) and asks `Proceed? [y/N]` before
  any VM is created. `--dry-run` exits after the estimate; `--yes` skips
  the prompt.
- **Spot preemption-safe.** Results are written per job; on preemption a
  re-launch picks up where it left off.
- **Original + normalized text preserved.** Every clip records both the
  raw and normalized reference + hypothesis, so post-hoc punctuation /
  casing comparisons are possible without re-running the benchmark.

## Prerequisites

1. `gcloud` installed and authenticated, with access to the
   `open-testimony` project.
2. `gsutil` (ships with `gcloud`).
3. The Compute Engine API enabled in `open-testimony` (it already is).
4. A GCS bucket name — defaults to `open-testimony-speechbench`, created
   automatically on first launch.
5. (Optional) `DASHSCOPE_API_KEY` set if you want to run the
   `qwen3.5-omni-*` models.
6. (Optional) HuggingFace token in `~/.cache/huggingface/token` if you want
   to use the gated `gigaspeech_l` or `spgispeech` datasets.

Install the orchestrator locally:

```bash
cd ~/experiments/speechbench
python -m pip install -e .
```

This installs `click`, `google-cloud-storage`, `tabulate`, `tqdm`, and
`pyyaml` — the heavy ML dependencies are only installed on the GPU VMs.

## Quick start

```bash
# 1. See what's available
speechbench list

# 2. Print a cost / wall-time estimate (no VMs created)
speechbench estimate --max-vms 8 --quick

# 3. Sanity-check launch — small per-dataset caps, fan out to 6 VMs
#    across 6 regions. Will print the estimate and prompt for confirmation.
speechbench launch \
    --run-id sanity \
    --models 'whisper-tiny.en' --models 'parakeet-tdt-0.6b-v2' --models 'gemma-4-E4B-it' \
    --datasets librispeech_clean --datasets ami_ihm \
    --gpu t4 --max-vms 6 --quick

# 4. Watch progress
speechbench status --run-id sanity

# 5. Pull results + render report once VMs have shut down
speechbench report --run-id sanity
```

## The full sweep

```bash
# Plan it (no VMs):
speechbench estimate --max-vms 8

# Run it: 23 local models × 6 ungated datasets = 138 jobs across 8 spot
# T4s in 8 different regions. Estimate is around 3.8h wall + ~$2.22.
speechbench launch --run-id main --max-vms 8 --gpu t4

# Add the DashScope API models (separate billing — Alibaba):
speechbench launch --run-id main \
    --models 'qwen3.5-omni-flash' --models 'qwen3.5-omni-light' \
    --api-key DASHSCOPE_API_KEY="$DASHSCOPE_API_KEY"

# Add a new dataset later — only the new (model × dataset) combinations run:
speechbench launch --run-id main --datasets gigaspeech_l

# Force re-run a single model (e.g. after a runtime upgrade):
speechbench launch --run-id main --models whisper-large-v3-turbo --rerun

# Get the report
speechbench report --run-id main
```

The report writes `results/main/report.md`, `results/main/report.csv`,
and `results/main/summary.json`, and prints a per-dataset table to the
terminal.

## Adding a new model

Edit `speechbench/models.py` and append an entry to `MODELS`. For most
HuggingFace ASR models you only need to fill in `key`, `family`, `backend`,
`hf_id`, `min_vram_gb`, and rough `sec_per_audio_sec` / `load_seconds`
priors (these only feed the cost estimator). Re-launch with the same
`--run-id` and only the new model's jobs will run.

## Adding a new dataset

Edit `speechbench/datasets.py` and append an entry to `DATASETS`. The
default loader handles any HF dataset that exposes an `audio` column and
a text column — you only need a custom loader for unusual formats.
Re-launch with the same `--run-id` and only the new dataset's jobs will
run.

## Notes / caveats

- Model load and inference throughput estimates in `cost.py` are static
  priors with a 1.4× safety factor. They are usually conservative on a T4
  but you should treat the printed wall-time estimate as a ceiling, not an
  oracle.
- Spot VMs in some regions occasionally hit no-capacity errors. The
  orchestrator launches each VM individually so a single-region failure
  doesn't take down the rest of the run; re-launch with the same `--run-id`
  to retry.
- Whisper-style normalization is light by design (lowercase, contractions,
  punctuation strip). For deeper analysis use the per-clip
  `reference_raw` / `hypothesis_raw` fields in each `results/*.json`.
- DashScope (Qwen3.5-Omni) calls are billed separately by Alibaba. We do
  not include their cost in the spot $ estimate; the API model jobs are
  scheduled like any other but the GPU on the VM is unused for them.

## Layout

```
speechbench/
├── speechbench/                  # python package
│   ├── cli.py                    # `speechbench` entry point
│   ├── config.py                 # GCP defaults, GPU specs
│   ├── models.py                 # ModelSpec + ASRModel wrappers + REGISTRY
│   ├── datasets.py               # DatasetSpec + loaders + REGISTRY
│   ├── eval.py                   # normalizer, WER/CER, GPU memory sampler
│   ├── cost.py                   # static cost / wall-time estimator
│   ├── orchestrate.py            # local: build jobs, plan, partition, upload, launch
│   ├── runner.py                 # VM-side: load each model once, run all its jobs
│   ├── gcp.py                    # gcloud / gsutil wrappers
│   ├── report.py                 # local: pull results, render reports
│   └── startup.sh                # VM startup script (DLVM image)
├── requirements-vm.txt           # VM-side heavy deps (torch, nemo, transformers, …)
├── pyproject.toml                # local install (click + gcs + tabulate)
├── asr_benchmark_plan.md         # design doc
└── README.md                     # this file
```
