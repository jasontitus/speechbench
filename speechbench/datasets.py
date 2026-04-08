"""Dataset registry and loaders.

Each dataset is described by a `DatasetSpec`. The `load()` method returns an
iterator of (audio_id, np.ndarray, reference_text, audio_seconds) tuples,
already resampled to 16 kHz mono.

Adding a new dataset is one entry in `DATASETS` plus, if the loader is unusual,
a small loader function. The default `_default_loader` works for any HF dataset
that exposes an `audio` column and a text column.

This module is imported on the GPU VM, so it depends on `datasets`, `librosa`,
and `numpy`. Local-only paths (`speechbench launch`) read DATASETS but never
call `load()`, so they don't need those installed.
"""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Optional, Tuple

# numpy is heavy enough that we lazy-import it. The orchestrator process
# (laptop) reads DATASETS for cost estimation and globbing without needing
# numpy/datasets/librosa installed.
_np = None


def _lazy_np():
    global _np
    if _np is None:
        _np = importlib.import_module("numpy")
    return _np


# (audio_id, audio_array_16k_mono, reference_text, audio_seconds_float)
DatasetSample = Tuple[str, "object", str, float]


@dataclass(frozen=True)
class DatasetSpec:
    key: str  # short name used on the CLI and in job IDs
    hf_dataset: str  # HF datasets repo
    hf_config: Optional[str]  # HF dataset config name (None if not used)
    split: str  # HF split
    text_field: str  # column with the reference transcript
    audio_field: str = "audio"
    default_cap: int = 200  # default max samples (overridden by --full)
    full_size: Optional[int] = None  # informational
    requires_auth: bool = False  # gated dataset → may need HF token
    description: str = ""
    loader: Optional[Callable] = None  # override the default loader
    # Language hint used by multilingual models (Whisper, Gemma 4, etc.) to
    # set the right decoder prefix. For English-only datasets this stays
    # "english" (the default). For Spanish datasets: "spanish".
    language: str = "english"

    def make_job_id_part(self, sample_cap: int) -> str:
        return f"{self.key}@{sample_cap}"


def _default_loader(spec: DatasetSpec, sample_cap: int) -> Iterator[DatasetSample]:
    """Generic loader for HF datasets that have an `audio` and text column."""
    from datasets import load_dataset  # type: ignore
    import librosa  # type: ignore

    np = _lazy_np()

    kwargs = {}
    # Use streaming so we never download a whole dataset just to take 200 clips
    kwargs["streaming"] = True

    if spec.hf_config:
        ds = load_dataset(spec.hf_dataset, spec.hf_config, split=spec.split, **kwargs)
    else:
        ds = load_dataset(spec.hf_dataset, split=spec.split, **kwargs)

    n = 0
    for i, sample in enumerate(ds):
        if sample_cap > 0 and n >= sample_cap:
            break

        audio_obj = sample[spec.audio_field]
        if isinstance(audio_obj, dict):
            arr = audio_obj.get("array")
            sr = audio_obj.get("sampling_rate", 16000)
            path = audio_obj.get("path", "")
        else:
            # Some streaming datasets give a Audio() decoded dict; fall back.
            arr = audio_obj["array"]
            sr = audio_obj["sampling_rate"]
            path = audio_obj.get("path", "")

        if arr is None:
            continue

        arr = np.asarray(arr, dtype="float32")
        if arr.ndim == 2:
            arr = arr.mean(axis=1).astype("float32")
        if sr != 16000:
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16000).astype("float32")

        ref = sample.get(spec.text_field) or ""
        if not isinstance(ref, str):
            ref = str(ref)
        if not ref.strip():
            continue

        audio_seconds = float(len(arr) / 16000.0)
        sample_id = sample.get("id") or sample.get("audio_id") or sample.get("file_id") or path or f"{spec.key}-{i}"
        yield (str(sample_id), arr, ref, audio_seconds)
        n += 1


# ─── Dataset registry ──────────────────────────────────────────────────────────

DATASETS: dict[str, DatasetSpec] = {
    "librispeech_clean": DatasetSpec(
        key="librispeech_clean",
        hf_dataset="openslr/librispeech_asr",
        hf_config="clean",
        split="test",
        text_field="text",
        default_cap=500,
        full_size=2620,
        description="LibriSpeech test-clean — clean read speech.",
    ),
    "librispeech_other": DatasetSpec(
        key="librispeech_other",
        hf_dataset="openslr/librispeech_asr",
        hf_config="other",
        split="test",
        text_field="text",
        default_cap=500,
        full_size=2939,
        description="LibriSpeech test-other — harder read speech.",
    ),
    "ami_ihm": DatasetSpec(
        key="ami_ihm",
        hf_dataset="edinburghcstr/ami",
        hf_config="ihm",
        split="test",
        text_field="text",
        default_cap=200,
        full_size=12643,
        description="AMI IHM test — meeting/conversation, headset mic.",
    ),
    "earnings22": DatasetSpec(
        key="earnings22",
        hf_dataset="distil-whisper/earnings22",
        hf_config="chunked",
        split="test",
        text_field="transcription",
        default_cap=200,
        full_size=57516,
        description="Earnings22 — earnings call recordings (chunked ~20s).",
    ),
    "tedlium_longform": DatasetSpec(
        key="tedlium_longform",
        hf_dataset="distil-whisper/tedlium-long-form",
        hf_config=None,
        split="test",
        text_field="text",
        default_cap=11,
        full_size=11,
        description="TED-LIUM long-form — long-form lectures (11 speakers).",
    ),
    "voxpopuli_en": DatasetSpec(
        key="voxpopuli_en",
        hf_dataset="facebook/voxpopuli",
        hf_config="en",
        split="test",
        text_field="normalized_text",
        default_cap=300,
        description="VoxPopuli — European parliament speeches (English).",
    ),
    "gigaspeech_l": DatasetSpec(
        key="gigaspeech_l",
        hf_dataset="speechcolab/gigaspeech",
        hf_config="l",
        split="test",
        text_field="text",
        default_cap=300,
        requires_auth=True,
        description="GigaSpeech L — multi-domain (audiobooks/podcasts/YouTube). Gated.",
    ),
    "spgispeech": DatasetSpec(
        key="spgispeech",
        hf_dataset="kensho/spgispeech",
        hf_config="s",
        split="test",
        text_field="transcript",
        default_cap=300,
        requires_auth=True,
        description="SPGISpeech — financial filings audio. Gated.",
    ),
    # ─── Spanish ──────────────────────────────────────────────────────────
    "mls_es": DatasetSpec(
        key="mls_es",
        hf_dataset="facebook/multilingual_librispeech",
        hf_config="spanish",
        split="test",
        text_field="transcript",
        default_cap=300,
        language="spanish",
        description="Multilingual LibriSpeech — Spanish test set (audiobooks).",
    ),
    "voxpopuli_es": DatasetSpec(
        key="voxpopuli_es",
        hf_dataset="facebook/voxpopuli",
        hf_config="es",
        split="test",
        text_field="normalized_text",
        default_cap=300,
        language="spanish",
        description="VoxPopuli — European parliament speeches in Spanish.",
    ),
    "fleurs_es": DatasetSpec(
        key="fleurs_es",
        hf_dataset="google/fleurs",
        hf_config="es_419",
        split="test",
        text_field="transcription",
        default_cap=300,
        language="spanish",
        description="FLEURS Spanish (es_419) — 102-language multilingual ASR benchmark.",
    ),
    "common_voice_17_es": DatasetSpec(
        key="common_voice_17_es",
        hf_dataset="mozilla-foundation/common_voice_17_0",
        hf_config="es",
        split="test",
        text_field="sentence",
        default_cap=300,
        language="spanish",
        requires_auth=True,
        description="Common Voice 17 — Spanish. Crowd-sourced read speech. Gated (needs HF token).",
    ),
}


def list_dataset_keys() -> list[str]:
    return list(DATASETS.keys())


def load(spec: DatasetSpec, sample_cap: int) -> Iterator[DatasetSample]:
    loader = spec.loader or _default_loader
    return loader(spec, sample_cap)


def resolve(patterns: Iterable[str]) -> list[str]:
    """Resolve dataset glob patterns against DATASETS keys.

    Returns a deduplicated, ordered list of keys. Unknown patterns raise.
    """
    import fnmatch

    out: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        matches = [k for k in DATASETS if fnmatch.fnmatchcase(k, p)]
        if not matches:
            raise ValueError(f"No dataset matches pattern: {p!r}")
        for m in matches:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out
