"""Ad hoc CPU smoke test for parakeet-tdt-0.6b-v3.

Loads the model with NeMo on CPU (no CUDA) and transcribes a short
LibriSpeech sample. Prints load time, inference latency, RTFx, and the
transcript. This mirrors what speechbench/models.py::NeMoParakeetModel
already does on Mac (CPU) — the goal is to verify it also works on a
plain Linux CPU environment, not just on Apple Silicon.

Usage:
    python scripts/test_parakeet_cpu.py [path/to/audio.wav]

If no audio path is given, downloads the librosa 'libri1' sample
(~14.8 s, 16 kHz mono) and uses that.
"""

from __future__ import annotations

import os
import sys
import time

# Make sure no GPU is picked up even if one happens to be visible.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402


MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def _prepare_audio(path: str | None) -> str:
    if path and os.path.exists(path):
        return path
    # librosa ships a pooch-cached short LibriSpeech clip.
    src = librosa.ex("libri1", hq=False)
    y, sr = librosa.load(src, sr=16000, mono=True)
    out = "/tmp/parakeet_cpu_test.wav"
    sf.write(out, y, sr, format="WAV")
    return out


def main() -> int:
    audio_path = _prepare_audio(sys.argv[1] if len(sys.argv) > 1 else None)

    info = sf.info(audio_path)
    audio_s = info.frames / float(info.samplerate)
    print(f"audio: {audio_path}  dur={audio_s:.2f}s  sr={info.samplerate}")

    assert not torch.cuda.is_available(), "expected CPU-only environment"
    print(f"torch={torch.__version__}  cuda_available={torch.cuda.is_available()}")

    import nemo.collections.asr as nemo_asr  # type: ignore

    print(f"loading {MODEL_ID} on CPU ...")
    t0 = time.perf_counter()
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID)
    model = model.to("cpu")
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"load_s={load_s:.1f}")

    # Warm-up pass (first call does lazy initialisation and JIT work).
    print("warm-up transcribe ...")
    t0 = time.perf_counter()
    _ = model.transcribe([audio_path], batch_size=1, verbose=False)
    warm_s = time.perf_counter() - t0
    print(f"warm_s={warm_s:.2f}  warm_rtfx={audio_s / warm_s:.2f}")

    # Timed pass.
    print("timed transcribe ...")
    t0 = time.perf_counter()
    outs = model.transcribe([audio_path], batch_size=1, verbose=False)
    infer_s = time.perf_counter() - t0

    # NeMo's return type has shifted across versions; normalise to a string.
    item = outs[0] if outs else ""
    if isinstance(item, list):
        item = item[0] if item else ""
    if hasattr(item, "text"):
        item = item.text
    if isinstance(item, tuple):
        item = item[0]
    text = str(item).strip()

    rtfx = audio_s / infer_s if infer_s > 0 else 0.0
    print("--- result ---")
    print(f"audio_s      = {audio_s:.2f}")
    print(f"load_s       = {load_s:.1f}")
    print(f"warm_s       = {warm_s:.2f}")
    print(f"infer_s      = {infer_s:.2f}")
    print(f"rtfx         = {rtfx:.2f}x realtime")
    print(f"transcript   = {text!r}")
    return 0 if text else 1


if __name__ == "__main__":
    raise SystemExit(main())
