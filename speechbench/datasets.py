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
    # Some HF datasets (google/fleurs) ship a custom loading script that
    # needs trust_remote_code=True. Others (fsicoli/common_voice_22_0) ship
    # a script whose cast is broken — they work BETTER with
    # trust_remote_code=False (datasets library auto-detects parquet).
    # Default True (most datasets either need it or don't care).
    trust_remote_code: bool = True

    def make_job_id_part(self, sample_cap: int) -> str:
        return f"{self.key}@{sample_cap}"


def _common_voice_22_loader(spec: DatasetSpec, sample_cap: int) -> Iterator[DatasetSample]:
    """Custom loader for fsicoli/common_voice_22_0.

    The fsicoli mirror ships its data as `transcript/{lang}/test.tsv` +
    `audio/{lang}/test/{lang}_test_{shard}.tar` and includes a custom
    dataset script that the HF `datasets` library can no longer load
    reliably (schema cast mismatches on up_votes/down_votes columns).

    This loader bypasses the script entirely: it downloads the TSV, streams
    the tar file(s) with the test audio, extracts mp3s on the fly, and
    yields (clip_id, audio16k, sentence, seconds) tuples. The
    `requires_auth`/`trust_remote_code` flags don't apply because we're
    fetching raw files via `huggingface_hub.hf_hub_download`.
    """
    import csv as _csv
    import tarfile
    import tempfile
    import os as _os

    from huggingface_hub import hf_hub_download  # type: ignore
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore

    np = _lazy_np()

    lang = spec.hf_config or "en"
    split = spec.split

    # 1. Grab the TSV (tiny — typically < 1 MB).
    tsv_path = hf_hub_download(
        repo_id=spec.hf_dataset,
        repo_type="dataset",
        filename=f"transcript/{lang}/{split}.tsv",
    )

    # 2. Read TSV → list of (path, sentence) pairs. Keep order so the
    # sample_cap slice is deterministic across runs.
    rows: list[tuple[str, str]] = []
    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f, delimiter="\t")
        for r in reader:
            path = r.get("path") or ""
            sent = r.get("sentence") or r.get("sentence_raw") or ""
            if path and sent:
                rows.append((path, sent))
            if sample_cap and len(rows) >= sample_cap * 4:
                # Read a few × cap so we can still fill after skipping
                # any rows whose audio is missing. 4× is plenty in practice.
                break

    # 3. Build a lookup of path → row index for the rows we want.
    wanted = {p for p, _ in rows}

    # 4. Download the first tar shard and extract only the audio files we
    # care about. (In practice the test split is always one shard; if not,
    # we walk shards until we've found everything.)
    shard_idx = 0
    found: dict[str, tuple[str, str]] = {}  # path -> (tmp_path, sentence)
    sentence_by_path = {p: s for p, s in rows}
    tmpdir = tempfile.mkdtemp(prefix="cv22_")
    while len(found) < min(len(rows), sample_cap * 2 if sample_cap else len(rows)):
        try:
            tar_path = hf_hub_download(
                repo_id=spec.hf_dataset,
                repo_type="dataset",
                filename=f"audio/{lang}/{split}/{lang}_{split}_{shard_idx}.tar",
            )
        except Exception:
            break  # No more shards.
        with tarfile.open(tar_path, mode="r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = _os.path.basename(member.name)
                if name in wanted and name not in found:
                    tf.extract(member, path=tmpdir, set_attrs=False)
                    found[name] = (_os.path.join(tmpdir, member.name), sentence_by_path[name])
                    if sample_cap and len(found) >= sample_cap * 2:
                        break
        shard_idx += 1
        if shard_idx > 20:  # safety valve
            break

    # 5. Yield samples in TSV order, decoding mp3 → 16 kHz float32.
    n = 0
    for path, sentence in rows:
        if sample_cap and n >= sample_cap:
            break
        if path not in found:
            continue
        mp3_path, _ = found[path]
        try:
            audio, sr = librosa.load(mp3_path, sr=16000, mono=True)
        except Exception:
            try:
                audio, sr = sf.read(mp3_path)
                if sr != 16000:
                    audio = librosa.resample(
                        audio.astype("float32"), orig_sr=sr, target_sr=16000
                    )
            except Exception:
                continue
        if audio is None or len(audio) == 0:
            continue
        duration = float(len(audio) / 16000.0)
        yield (path, np.asarray(audio, dtype="float32"), sentence, duration)
        n += 1


def _default_loader(spec: DatasetSpec, sample_cap: int) -> Iterator[DatasetSample]:
    """Generic loader for HF datasets that have an `audio` and text column."""
    from datasets import load_dataset  # type: ignore
    import librosa  # type: ignore

    np = _lazy_np()

    kwargs = {}
    # Use streaming so we never download a whole dataset just to take 200 clips
    kwargs["streaming"] = True
    # Per-dataset opt-in. google/fleurs needs it True; fsicoli/common_voice_22_0
    # needs it False (their script's cast schema doesn't match the parquet).
    kwargs["trust_remote_code"] = getattr(spec, "trust_remote_code", True)

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
    "common_voice_22_en": DatasetSpec(
        key="common_voice_22_en",
        hf_dataset="fsicoli/common_voice_22_0",
        trust_remote_code=False,
        loader=lambda spec, cap: _common_voice_22_loader(spec, cap),
        hf_config="en",
        split="test",
        text_field="sentence",
        default_cap=300,
        language="english",
        description="Common Voice 22 — English. Crowd-sourced read speech "
                    "(diverse speakers/accents). fsicoli mirror on HF.",
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
    "common_voice_22_es": DatasetSpec(
        key="common_voice_22_es",
        hf_dataset="fsicoli/common_voice_22_0",
        trust_remote_code=False,
        loader=lambda spec, cap: _common_voice_22_loader(spec, cap),
        hf_config="es",
        split="test",
        text_field="sentence",
        default_cap=300,
        language="spanish",
        description="Common Voice 22 — Spanish. Crowd-sourced read speech. "
                    "fsicoli mirror (Mozilla moved official CV off HF in Oct 2025).",
    ),
    # ─── Lithuanian ───────────────────────────────────────────────────────
    "voxpopuli_lt": DatasetSpec(
        key="voxpopuli_lt",
        hf_dataset="facebook/voxpopuli",
        hf_config="lt",
        split="test",
        text_field="normalized_text",
        default_cap=300,
        language="lithuanian",
        description="VoxPopuli — European parliament speeches in Lithuanian.",
    ),
    "fleurs_lt": DatasetSpec(
        key="fleurs_lt",
        hf_dataset="google/fleurs",
        hf_config="lt_lt",
        split="test",
        text_field="transcription",
        default_cap=300,
        language="lithuanian",
        description="FLEURS Lithuanian (lt_lt) — 102-language multilingual ASR benchmark.",
    ),
    "common_voice_22_lt": DatasetSpec(
        key="common_voice_22_lt",
        hf_dataset="fsicoli/common_voice_22_0",
        trust_remote_code=False,
        loader=lambda spec, cap: _common_voice_22_loader(spec, cap),
        hf_config="lt",
        split="test",
        text_field="sentence",
        default_cap=300,
        language="lithuanian",
        description="Common Voice 22 — Lithuanian. Crowd-sourced read speech.",
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
