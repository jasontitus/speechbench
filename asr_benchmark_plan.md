# speechbench — ASR Benchmark Suite on GCP Spot GPUs

## Goal
Evaluate a wide range of ASR models on a wide range of standard benchmarks,
on **GCP spot GPU instances** in the `open-testimony` project, and produce a
single comparative report covering **WER**, **CER**, **RTFx** (real-time
factor), **GPU peak memory**, and **per-clip latency**.

The benchmark must be one-command launchable from a laptop. The orchestrator
spins up one or more spot GPU VMs, each VM downloads the dataset and the
benchmark code, runs its assigned (model × dataset) jobs, uploads per-job JSON
to a GCS bucket, and shuts itself down. A local `report` command then
aggregates the JSONs into a Markdown / CSV / JSON comparative report.

A core design requirement: **runs are additive**. Adding a new dataset (or a
new model) must NOT re-run combinations that already have results. The runner
checks GCS for an existing result file with the same deterministic job ID and
skips it; only `--rerun` forces re-execution.

---

## Datasets

All loaded via the `datasets` library. The dataset registry lives in
`speechbench/datasets.py` — adding a dataset is one entry. Default sample
caps are conservative; pass `--full` for the entire test split.

| Key                  | HF dataset                          | Config | Split | Text field      | Default cap | Notes                          |
| -------------------- | ----------------------------------- | ------ | ----- | --------------- | ----------- | ------------------------------ |
| `librispeech_clean`  | `openslr/librispeech_asr`           | clean  | test  | `text`          | 500         | Standard read speech           |
| `librispeech_other`  | `openslr/librispeech_asr`           | other  | test  | `text`          | 500         | Harder read speech             |
| `ami_ihm`            | `edinburghcstr/ami`                 | ihm    | test  | `text`          | 200         | Meetings, headset mic          |
| `earnings22`         | `distil-whisper/earnings22`         | chunked| test  | `transcription` | 200         | Earnings calls (~20s chunks)   |
| `tedlium_longform`   | `distil-whisper/tedlium-long-form`  | -      | test  | `text`          | 11 (all)    | Long-form lectures             |
| `voxpopuli_en`       | `facebook/voxpopuli`                | en     | test  | `normalized_text`| 300        | Parliamentary speech           |
| `gigaspeech_l`       | `speechcolab/gigaspeech`            | l      | test  | `text`          | 300         | Gated — requires HF auth       |
| `spgispeech`         | `kensho/spgispeech`                 | s      | test  | `transcript`    | 300         | Gated — requires HF auth       |

Audio is resampled to 16 kHz mono by the loader if it isn't already.
Gated datasets (gigaspeech, spgispeech) gracefully skip with a warning if
HF credentials are missing.

**Adding new datasets** is meant to be easy. The user appends an entry to
`DATASETS` in `speechbench/datasets.py` (or via a YAML override file at
`~/.speechbench/datasets.yaml`) and re-runs `speechbench launch` with the
same `--run-id` — only the new (model × new dataset) combinations run.

---

## Models

The runner is a thin wrapper over a model registry. Adding a new HF model is
one entry in `speechbench/models.py`.

### Whisper family — HF Transformers + faster-whisper

| Registry key                       | Backend         |
| ---------------------------------- | --------------- |
| `whisper-tiny.en`                  | transformers    |
| `whisper-base.en`                  | transformers    |
| `whisper-small.en`                 | transformers    |
| `whisper-medium.en`                | transformers    |
| `whisper-large-v2`                 | transformers    |
| `whisper-large-v3`                 | transformers    |
| `whisper-large-v3-turbo`           | transformers    |
| `distil-large-v3`                  | transformers    |
| `fw-large-v3`                      | faster-whisper  |
| `fw-large-v3-turbo`                | faster-whisper  |
| `fw-distil-large-v3`               | faster-whisper  |

Both backends are exposed so we can compare HF FP16 vs CTranslate2 INT8 on
the same GPU.

### Parakeet family — NeMo

| Registry key                       |
| ---------------------------------- |
| `parakeet-tdt-0.6b-v3`             |
| `parakeet-tdt-0.6b-v2`             |
| `parakeet-tdt-1.1b`                |
| `parakeet-rnnt-1.1b`               |
| `parakeet-rnnt-0.6b`               |
| `parakeet-ctc-1.1b`                |
| `parakeet-ctc-0.6b`                |
| `parakeet-tdt_ctc-110m`            |

### Qwen ASR

| Registry key                       | Backend         | Notes                                         |
| ---------------------------------- | --------------- | --------------------------------------------- |
| `qwen3-asr-0.6b`                   | transformers    | `Qwen/Qwen3-ASR-0.6B`, runs on T4             |
| `qwen3-asr-1.7b`                   | transformers    | `Qwen/Qwen3-ASR-1.7B`, runs on T4             |
| `qwen3.5-omni-plus`                | dashscope-api   | DashScope API model (no local weights)        |
| `qwen3.5-omni-flash`               | dashscope-api   | DashScope API model                           |
| `qwen3.5-omni-light`               | dashscope-api   | DashScope API model                           |

The Qwen3.5-Omni weights are **not published on HuggingFace** as of this
writing — Alibaba ships them only via the DashScope API. The runner has a
`DashScopeOmniModel` class that hits the OpenAI-compatible endpoint at
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. The user must set
`DASHSCOPE_API_KEY` and pass `--api-keys` to the launcher; the orchestrator
forwards keys to VM metadata. API-model inference happens from the same VM as
everything else (no GPU needed for that job, but co-locating keeps the runner
simple).

### Gemma 4 (multimodal LLM, prompt-based transcription)

| Registry key                       | Backend         | Notes                                         |
| ---------------------------------- | --------------- | --------------------------------------------- |
| `gemma-4-E4B-it`                   | transformers    | `google/gemma-4-E4B-it`, ~8B raw / 4.5B eff.  |
| `gemma-4-E2B-it`                   | transformers    | `google/gemma-4-E2B-it`, smaller variant      |

Gemma 4 E2B/E4B-it are multimodal LLMs with native audio understanding via a
USM-style conformer encoder. We do transcription by prompting:
`"Transcribe the audio exactly as spoken. Output only the transcription."`
The model class is `AutoModelForMultimodalLM`, the processor is
`AutoProcessor`, and inputs are built with the chat template (audio as a
content block). E4B at FP16 fits in ~9 GB so a T4 (16 GB) is sufficient.
Long clips are chunked at 28 s with 0.5 s overlap (Gemma's audio context is
limited) and the chunk transcripts are joined with a space — same approach as
the original Mac/Ollama plan, but running natively on the GPU VM via HF
Transformers instead of via Ollama. This means we get apples-to-apples GPU
metrics for Gemma alongside the Whisper / Parakeet / Qwen models.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │  Local laptop                            │
                    │  ┌────────────────────────────────────┐  │
                    │  │  speechbench launch                │  │
                    │  │   • build job list (M × D)         │  │
                    │  │   • diff against existing results  │  │
                    │  │   • tar source → GCS               │  │
                    │  │   • create N spot GPU VMs          │  │
                    │  │     (instance metadata = job set)  │  │
                    │  └────────────────────────────────────┘  │
                    │  ┌────────────────────────────────────┐  │
                    │  │  speechbench status / report       │  │
                    │  │   • pull all */result.json from    │  │
                    │  │     GCS, build Markdown/CSV table  │  │
                    │  └────────────────────────────────────┘  │
                    └─────────────────┬────────────────────────┘
                                      │ gcloud compute / gsutil
                    ┌─────────────────┴────────────────────────┐
                    │  GCS bucket: gs://open-testimony-speechbench/
                    │   ├─ src/speechbench-<run_id>.tar.gz
                    │   └─ runs/<run_id>/
                    │       ├─ jobs.json          (full work list)
                    │       ├─ assignments/<vm>.json
                    │       ├─ results/<job_id>.json
                    │       └─ logs/<vm_name>.log
                    └─────────────────┬────────────────────────┘
                                      │
       ┌──────────────────────────────┼───────────────────────────────┐
       │                              │                               │
┌──────┴──────┐               ┌───────┴─────┐                  ┌──────┴──────┐
│  spot VM 1  │               │  spot VM 2  │   …              │  spot VM N  │
│  T4 / L4    │               │  T4 / L4    │                  │  T4 / L4    │
│  startup.sh │               │  startup.sh │                  │  startup.sh │
│   • install │               │   • install │                  │   • install │
│   • run job │               │   • run job │                  │   • run job │
│   • upload  │               │   • upload  │                  │   • upload  │
│   • shutdown│               │   • shutdown│                  │   • shutdown│
└─────────────┘               └─────────────┘                  └─────────────┘
```

### Key design points

1. **Job = (model_key, dataset_key, sample_cap)**. Jobs are independent and
   addressable by a deterministic `job_id = sha1("model|dataset|cap")[:12]`.
2. **Additive runs.** Re-running `speechbench launch` with the same
   `--run-id` checks GCS for each prospective `job_id`; existing results are
   skipped. Adding a new dataset and re-launching only runs the new
   combinations. `--rerun` forces re-execution.
3. **Spot preemption.** VMs use
   `--provisioning-model=SPOT --instance-termination-action=DELETE`. The
   runner uploads results after every job, so a preemption mid-run loses at
   most one job — re-launching with the same `--run-id` resumes cleanly.
4. **Per-VM job batching.** Default packing is "fewest VMs": one VM is given
   the entire sweep. `--max-vms N` fans out across N VMs (round-robin job
   slicing). The default of 1 stays inside the open-testimony quota of 1
   spot T4/L4 in `us-central1` — the orchestrator queries that quota at
   launch time and refuses to exceed it.
5. **One bucket per project, one prefix per run.** Run IDs are timestamps
   unless explicitly named so reports stay tied to the launch that produced
   them. A persistent name like `--run-id main` is the recommended pattern
   for accumulating results across many launches.

---

## Project layout

```
speechbench/
├── asr_benchmark_plan.md           # this file
├── README.md                       # quick start
├── pyproject.toml                  # local install (orchestrator + runner)
├── requirements-vm.txt             # VM-side dependencies (heavy: torch, nemo, …)
├── speechbench/
│   ├── __init__.py
│   ├── cli.py                      # `speechbench` entry point
│   ├── config.py                   # GCP defaults, bucket name, GPU choice
│   ├── models.py                   # ASRModel base class + all wrappers + REGISTRY
│   ├── datasets.py                 # dataset loaders (HF) + DATASETS registry
│   ├── eval.py                     # whisper-style normalizer + jiwer WER + metrics
│   ├── runner.py                   # VM-side: take an assignment, run jobs, upload
│   ├── gcp.py                      # gcloud / gsutil wrappers
│   ├── orchestrate.py              # local: build jobs, tar src, launch VMs
│   ├── report.py                   # local: pull results, render comparative report
│   └── startup.sh                  # template VM startup script
└── scripts/
    └── launch.sh                   # convenience wrapper for the most common case
```

---

## Metrics

```python
# eval.py — what we measure per (model, dataset)
{
  "job_id": "ab12cd34ef56",
  "model_key": "whisper-large-v3-turbo",
  "model_id": "openai/whisper-large-v3-turbo",
  "backend": "transformers",
  "dataset_key": "ami_ihm",
  "num_clips": 200,
  "wer": 0.184,
  "cer": 0.092,
  "rtfx_mean": 22.4,
  "rtfx_p50": 23.1,
  "rtfx_p10": 9.8,
  "latency_ms_mean": 380,
  "latency_ms_p50": 360,
  "latency_ms_p90": 720,
  "gpu_peak_mem_mb": 4820,         # via pynvml, sampled mid-inference
  "model_load_mem_mb": 3120,
  "audio_duration_s_total": 6800.4,
  "wall_time_s": 304.1,
  "gpu_name": "Tesla T4",
  "gpu_compute_capability": "7.5",
  "torch_dtype": "float16",
  "failed_clips": 0,
  "clips": [ {id, ref, hyp, wer, cer, latency_ms, audio_s}, ... ]
}
```

**Normalization** uses a Whisper-style English text normalizer (lowercase,
punctuation strip, contraction expansion) so we get a consistent comparison
across models that emit punctuation/case differently. CER is computed on the
same normalized text.

**GPU memory** is captured by polling `nvmlDeviceGetMemoryInfo` from a
background thread at 100 ms intervals during inference; we report the
high-water mark above the post-load baseline.

---

## Orchestrator behavior (`speechbench launch`)

```
$ speechbench launch \
    --run-id main \
    --models 'whisper-*' parakeet-tdt-0.6b-v3 qwen3-asr-0.6b \
    --datasets librispeech_clean ami_ihm earnings22 \
    --gpu t4 \
    --max-vms 1 \
    --quick                            # 50 / 20 cap for sanity check
```

Steps:

1. Resolve `--models` and `--datasets` globs against the registries.
2. Build the job matrix `[(m, d) for m in models for d in datasets]`.
3. Compute `job_id` for each, then list `gs://…/runs/<run_id>/results/` and
   skip jobs whose `job_id.json` already exists (unless `--rerun`).
4. Print the diff: "X jobs already done, Y jobs to run". On `--dry-run`, exit.
5. Query the regional quota for the chosen GPU type → cap `--max-vms`.
6. Create the GCS bucket if absent. Generate or reuse `run_id`.
7. Tar and upload the source → `gs://.../src/<run_id>.tar.gz` (or reuse).
8. For each VM slot, write per-VM `assignments/<vm>.json` and
   `gcloud compute instances create`:
   - Image: `deeplearning-platform-release/pytorch-2-7-cu128-ubuntu-2204-nvidia-570`
   - Machine: `n1-standard-4` (T4) or `g2-standard-8` (L4)
   - Accelerator: 1 × T4 or 1 × L4
   - Spot, delete-on-termination
   - Metadata: `run-id`, `assignment-uri`, `bucket`, `startup-script` (file)
9. Print a `gcloud compute ssh` line, the GCS log path, and a hint to run
   `speechbench status` and `speechbench report`.

`speechbench status --run-id main` polls the bucket and prints which jobs
are done / running / missing. `speechbench report --run-id main` pulls
every `results/*.json` and emits the comparative report.

---

## VM startup script (`startup.sh`)

The DLVM image already has CUDA + a usable Python. The startup script:

1. Activates the base conda env shipped with the image.
2. Installs `requirements-vm.txt` (faster-whisper, transformers, NeMo, jiwer,
   datasets, pynvml, openai-sdk for DashScope, …) — pinned for reproducibility.
3. Downloads `src/<run_id>.tar.gz` and extracts it to `/opt/speechbench`.
4. Downloads its `assignments/<vm>.json` from GCS.
5. Runs `python -m speechbench.runner --assignment assignment.json
    --bucket gs://… --run-id … 2>&1 | tee /var/log/speechbench.log`.
6. Uploads `/var/log/speechbench.log` to `gs://…/runs/<run_id>/logs/<vm>.log`.
7. `shutdown -h now`.

If a step fails the script writes a `.failed` marker and the VM still
shuts itself down so we don't burn money on stuck instances.

---

## Resume + retry

- `speechbench launch --run-id <existing>` reuses the bucket prefix and the
  source tarball, recomputes the missing jobs, and only launches VMs for the
  remaining work.
- Spot preemptions are handled the same way — re-launch with the same
  `--run-id` and only the missing jobs are picked up.
- `speechbench launch --run-id <existing> --rerun` forces re-execution of
  the listed (model × dataset) combinations even if results exist.
- `speechbench launch --run-id <existing> --datasets <new_only>` is the
  intended pattern for "I added a new dataset, run only that".

---

## Reporting (`speechbench report`)

```
$ speechbench report --run-id main

▶ Pulled 24/24 result files from gs://open-testimony-speechbench/runs/main/results/
▶ Wrote results/main/report.md
▶ Wrote results/main/report.csv
▶ Wrote results/main/summary.json

================================================================================
SpeechBench — run main (T4 spot, us-central1-a)
================================================================================

LibriSpeech test-clean
Model                                       WER     CER    RTFx   GPU MB
parakeet-tdt-0.6b-v2                       1.86%   0.71%   175.4    2340
whisper-large-v3-turbo  (HF)               2.64%   1.00%    24.1    4820
whisper-large-v3        (HF)               2.71%   1.04%    11.3    5840
fw-large-v3-turbo       (CT2)              2.69%   1.02%    61.7    1920
…

AMI IHM test
Model                                       WER     CER    RTFx   GPU MB
…

Earnings22 test
Model                                       WER     CER    RTFx   GPU MB
…
```

Same data goes into:
- `report.md`  — human-readable Markdown with one table per dataset
- `report.csv` — flat CSV for importing into a sheet
- `summary.json` — machine-readable

---

## Cost / runtime envelope

- **T4 spot in us-central1**: ~$0.11/hr GPU + ~$0.04/hr instance.
- A full sweep (all Whisper + Parakeet + Qwen3-ASR on LS-clean+other, AMI
  IHM, Earnings22, TED-LIUM long-form, default sample caps) on a single T4
  should fit in ~6–10 hours, well under $2.
- Qwen3.5-Omni API calls are charged separately via DashScope.
- The orchestrator is built so that as the user's quota grows we can fan out
  to more VMs and shorten wall time linearly without code changes.

---

## Out of scope (for v1)

- Streaming / partial-result evaluation
- Diarization / speaker-attributed WER
- Multi-language WER (only English in v1; v3 Parakeets and large-v3 Whisper
  *can* do other languages but we are scoring with English refs)
- Comparing decoding hyperparameters (we use vendor defaults)
- Throughput benchmarks at >1 batch (we measure single-clip latency to keep
  the comparison apples-to-apples)
