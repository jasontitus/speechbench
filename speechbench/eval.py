"""Evaluation primitives: text normalization, WER/CER, GPU memory tracking."""
from __future__ import annotations

import re
import threading
import time
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional


# ─── Text normalization ────────────────────────────────────────────────────────
#
# Two normalizers live here:
#
#   • English: lowercase → expand contractions → strip punctuation → digit-to-
#     word for single digits → collapse whitespace. Roughly Whisper-style.
#
#   • Non-English: lowercase → strip everything that isn't a Unicode letter,
#     digit, apostrophe, or whitespace → collapse whitespace. We deliberately
#     keep all Unicode letters (Lithuanian ė, Spanish ñ, German ü, Cyrillic,
#     CJK, …) so morphologically rich languages don't lose word boundaries.
#
# Earlier versions of this file used a single ASCII-only regex
# (`[^a-z0-9'\s]`) for both, which silently corrupted every non-English
# transcript: every non-ASCII letter became a space, splitting words and
# inflating WER by ~2× on Lithuanian and ~1.3× on Spanish. The recompute path
# in this module re-derives historical WER/CER from per-clip raw fields, so
# old result JSONs can be fixed in place without touching a GPU.

_MULTI_WS_RE = re.compile(r"\s+")

_CONTRACTIONS = {
    "won't": "will not",
    "can't": "cannot",
    "n't": " not",
    "'re": " are",
    "'s": " is",
    "'d": " would",
    "'ll": " will",
    "'t": " not",
    "'ve": " have",
    "'m": " am",
}

_NUMBER_WORDS = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}


def _strip_punct_unicode(s: str) -> str:
    """Replace any character that isn't a Unicode letter/number, apostrophe,
    or whitespace with a space. Unicode-safe substitute for `[^a-z0-9'\\s]`."""
    out: list[str] = []
    for ch in s:
        if ch in (" ", "'", "\t", "\n"):
            out.append(ch)
            continue
        # Unicode general categories: L* = letter, N* = number/digit.
        cat = unicodedata.category(ch)
        if cat and cat[0] in ("L", "N"):
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def normalize_text(text: str, language: str = "english") -> str:
    """Normalize a string for fair WER comparison across ASR models.

    For English, the historical rules apply: lowercase, expand contractions,
    strip punctuation, single-digit→word, collapse whitespace.

    For every other language, we still lowercase + strip punctuation +
    collapse whitespace, but the punctuation strip uses Unicode categories
    instead of ASCII so non-Latin letters survive intact. We deliberately
    skip both the English contraction expansion (it would corrupt
    apostrophe-bearing tokens in other languages) and the digit-to-word
    substitution (only meaningful in English).
    """
    if text is None:
        return ""
    s = text.lower()
    if language == "english":
        for k, v in _CONTRACTIONS.items():
            s = s.replace(k, v)
        s = _strip_punct_unicode(s)
        # Spell out single isolated digits — multi-digit numbers stay as
        # numerals (the models are inconsistent and either choice is a wash
        # on the English leaderboard datasets).
        s = " ".join(_NUMBER_WORDS.get(tok, tok) for tok in s.split())
    else:
        s = _strip_punct_unicode(s)
    s = _MULTI_WS_RE.sub(" ", s).strip()
    return s


# ─── WER / CER ─────────────────────────────────────────────────────────────────


def compute_wer(
    references: list[str],
    hypotheses: list[str],
    language: str = "english",
) -> float:
    import jiwer  # type: ignore

    refs = [normalize_text(r, language=language) for r in references]
    hyps = [normalize_text(h, language=language) for h in hypotheses]
    refs = [r if r else " " for r in refs]
    hyps = [h if h else " " for h in hyps]
    return float(jiwer.wer(refs, hyps))


def compute_cer(
    references: list[str],
    hypotheses: list[str],
    language: str = "english",
) -> float:
    import jiwer  # type: ignore

    refs = [normalize_text(r, language=language) for r in references]
    hyps = [normalize_text(h, language=language) for h in hypotheses]
    refs = [r if r else " " for r in refs]
    hyps = [h if h else " " for h in hyps]
    return float(jiwer.cer(refs, hyps))


def per_clip_wer(reference: str, hypothesis: str, language: str = "english") -> float:
    return compute_wer([reference], [hypothesis], language=language)


def per_clip_cer(reference: str, hypothesis: str, language: str = "english") -> float:
    return compute_cer([reference], [hypothesis], language=language)


# ─── Latency / RTFx helpers ────────────────────────────────────────────────────


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# ─── GPU memory polling ────────────────────────────────────────────────────────


@dataclass
class GPUInfo:
    name: str = "unknown"
    compute_capability: str = ""
    total_mb: float = 0.0


class GPUMemorySampler:
    """Background thread that polls nvmlDeviceGetMemoryInfo every interval_ms."""

    def __init__(self, device_index: int = 0, interval_ms: int = 100):
        self.device_index = device_index
        self.interval_s = interval_ms / 1000.0
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._peak_mb: float = 0.0
        self._baseline_mb: float = 0.0
        self._available = False
        self._handle = None
        self._info = GPUInfo()

    def start(self) -> None:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            self._available = True

            try:
                self._info.name = pynvml.nvmlDeviceGetName(self._handle).decode() if isinstance(
                    pynvml.nvmlDeviceGetName(self._handle), bytes
                ) else pynvml.nvmlDeviceGetName(self._handle)
            except Exception:
                pass
            try:
                major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(self._handle)
                self._info.compute_capability = f"{major}.{minor}"
            except Exception:
                pass
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self._info.total_mb = mem.total / (1024 * 1024)
                self._baseline_mb = mem.used / (1024 * 1024)
                self._peak_mb = self._baseline_mb
            except Exception:
                pass

            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception:
            self._available = False

    def _loop(self) -> None:
        import pynvml  # type: ignore

        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                used_mb = mem.used / (1024 * 1024)
                if used_mb > self._peak_mb:
                    self._peak_mb = used_mb
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def reset_peak(self) -> None:
        try:
            import pynvml  # type: ignore

            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self._baseline_mb = mem.used / (1024 * 1024)
            self._peak_mb = self._baseline_mb
        except Exception:
            pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._thread = None
        try:
            import pynvml  # type: ignore

            pynvml.nvmlShutdown()
        except Exception:
            pass

    @property
    def info(self) -> GPUInfo:
        return self._info

    @property
    def peak_mb(self) -> float:
        return self._peak_mb

    @property
    def baseline_mb(self) -> float:
        return self._baseline_mb


# ─── Per-clip record ───────────────────────────────────────────────────────────


@dataclass
class ClipResult:
    clip_id: str
    audio_seconds: float
    reference_raw: str
    reference_norm: str
    hypothesis_raw: str
    hypothesis_norm: str
    latency_ms: float
    wer: float
    cer: float
    failed: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class JobResult:
    job_id: str
    model_key: str
    model_id: str
    backend: str
    family: str
    dataset_key: str
    sample_cap: int

    num_clips: int = 0
    failed_clips: int = 0
    audio_duration_s_total: float = 0.0
    wall_time_s: float = 0.0
    model_load_seconds: float = 0.0

    wer: float = 0.0
    cer: float = 0.0
    rtfx_mean: float = 0.0
    rtfx_p50: float = 0.0
    rtfx_p10: float = 0.0
    latency_ms_mean: float = 0.0
    latency_ms_p50: float = 0.0
    latency_ms_p90: float = 0.0

    gpu_name: str = ""
    gpu_compute_capability: str = ""
    gpu_total_mb: float = 0.0
    gpu_peak_mem_mb: float = 0.0
    model_load_mem_mb: float = 0.0

    started_at: float = 0.0
    finished_at: float = 0.0
    clips: list[ClipResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["clips"] = [c.to_dict() for c in self.clips]
        return d
