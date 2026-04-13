"""Model registry and wrappers.

Every ASR backend implements the `ASRModel` interface:
    load() → loads weights into GPU memory
    transcribe(audio, sr) → returns a raw transcript string
    unload()           → frees memory between models

Adding a model is one entry in `MODELS`. The orchestrator (running on the
laptop) only reads ModelSpec metadata; the heavy import + instantiation only
happens on the GPU VM inside `runner.py`.

The wrappers themselves do `import torch`, `import nemo_toolkit`, etc.
*lazily* — module import is cheap so the orchestrator can pull `MODELS` for
cost estimation + globbing without dragging in CUDA / NeMo / transformers.

Platform dispatch: `make_model(spec)` picks an MLX-based wrapper on Apple
Silicon (when the relevant `mlx-*` library is installed) and the standard
HF/NeMo wrapper everywhere else. The same registry key works on both —
`whisper-large-v3` runs via mlx-whisper on Mac and via HF Transformers on
Linux+CUDA.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import platform
import sys
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# ─── Platform / backend detection ──────────────────────────────────────────────


def is_mac_arm() -> bool:
    return sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")


def _have_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def have_mlx_whisper() -> bool:
    return is_mac_arm() and _have_module("mlx_whisper")


def have_mlx_qwen3_asr() -> bool:
    return is_mac_arm() and _have_module("mlx_qwen3_asr")


# ─── ModelSpec (registry entry — pure data, no heavy imports) ──────────────────


@dataclass(frozen=True)
class ModelSpec:
    key: str  # short name used on the CLI and in job IDs
    family: str  # whisper / parakeet / qwen3-asr / qwen3.5-omni / gemma4
    backend: str  # transformers / faster-whisper / nemo / dashscope-api
    hf_id: str  # HF repo or remote model name
    min_vram_gb: int  # rough lower bound on GPU memory needed
    description: str = ""
    # Optional PEFT adapter repo id or local path. When set, the model loader
    # applies the adapter on top of the base model (hf_id) after loading.
    adapter_id: str = ""
    # Static cost-model inputs (used by speechbench.cost):
    # rough wall seconds per *audio second* of input at FP16 on a T4.
    # 0.05 = 20× real-time. 1.0 = real-time. 2.0 = half real-time.
    sec_per_audio_sec: float = 0.1
    # one-time model load wall-time on a T4 (seconds), used to amortize.
    load_seconds: float = 60.0
    needs_dashscope_key: bool = False
    is_api: bool = False
    # Optional beam+LM decoding config. If set, the model wrapper applies
    # beam search with n-gram LM fusion after loading the model. The LM
    # file is downloaded from the model's HF repo (lm_hf_filename) or
    # loaded from a local path (lm_local_path).
    beam_size: int = 0  # 0 = greedy (default)
    lm_alpha: float = 0.5
    lm_hf_filename: str = ""  # e.g. "lt_domain_5gram.arpa"
    lm_local_path: str = ""  # fallback if HF download not available


# ─── Base class (used at runtime on the VM) ────────────────────────────────────


class ASRModel(ABC):
    spec: ModelSpec

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        """Transcribe an audio array.

        `language` is a full English name ("english", "spanish", …). Multilingual
        models use it to set the decoder prefix. English-only models (Whisper .en,
        parakeet English-only variants) ignore it.
        """

    def unload(self) -> None:
        gc.collect()
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                # Mac (Apple Silicon) — same housekeeping as ScribeBox
                torch.mps.empty_cache()
        except Exception:
            pass


# ─── Whisper via HF transformers ───────────────────────────────────────────────


class HFWhisperModel(ASRModel):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._pipe = None

    def load(self) -> None:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            pipeline,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.spec.hf_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        processor = AutoProcessor.from_pretrained(self.spec.hf_id)

        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=dtype,
            device=device,
            chunk_length_s=30,
            return_timestamps=False,
        )

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        # The HF ASR pipeline accepts a numpy array via {"array": ..., "sampling_rate": ...}.
        # English-only Whisper variants (".en") reject `language`/`task` kwargs.
        is_english_only = self.spec.hf_id.endswith(".en")
        gen_kwargs = {} if is_english_only else {"language": language, "task": "transcribe"}
        # Beam search if configured in the ModelSpec.
        if self.spec.beam_size > 0:
            gen_kwargs["num_beams"] = self.spec.beam_size
        out = self._pipe(
            {"array": audio, "sampling_rate": sample_rate},
            generate_kwargs=gen_kwargs,
        )
        return (out.get("text") or "").strip()

    def unload(self) -> None:
        self._pipe = None
        super().unload()


# ─── Whisper via mlx-whisper (Apple Silicon) ───────────────────────────────────


def _mlx_whisper_candidates(hf_id: str) -> list[str]:
    """Return a list of likely mlx-community repo IDs for an OpenAI Whisper id.

    The mlx-community naming is inconsistent — some repos use a `-mlx` suffix,
    some don't. We try the variants in order until one works, then cache the
    winner on the wrapper instance.
    """
    base = hf_id.split("/")[-1]  # whisper-tiny.en, whisper-large-v3-turbo, ...
    return [
        f"mlx-community/{base}",
        f"mlx-community/{base}-mlx",
        f"mlx-community/{base}-fp16",
    ]


class MLXWhisperModel(ASRModel):
    """Whisper via mlx-whisper (Apple Silicon)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._candidates = _mlx_whisper_candidates(spec.hf_id)
        self._mlx_repo: Optional[str] = None
        self._loaded = False

    def load(self) -> None:
        import numpy as np  # type: ignore
        import mlx_whisper  # type: ignore

        # 0.1 s of silence — forces the model fetch + compile, and lets us
        # discover which mlx-community repo variant exists for this base name.
        silent = np.zeros(1600, dtype="float32")
        last_err: Exception | None = None
        for repo in self._candidates:
            try:
                mlx_whisper.transcribe(silent, path_or_hf_repo=repo)
                self._mlx_repo = repo
                self._loaded = True
                return
            except Exception as e:
                last_err = e
                # Only retry on 404-ish errors; other errors mean the repo
                # exists but something else broke and we should surface it.
                msg = str(e).lower()
                if "404" in msg or "not found" in msg or "repositorynotfound" in msg:
                    continue
                raise
        raise RuntimeError(
            f"could not find an mlx-community variant for {self.spec.hf_id}; "
            f"tried {self._candidates}; last error: {last_err}"
        )

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        import mlx_whisper  # type: ignore

        # mlx-whisper uses 2-letter ISO codes: "en", "es", ...
        lang_code = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "lithuanian": "lt",
            "portuguese": "pt",
            "italian": "it",
            "dutch": "nl",
            "polish": "pl",
        }.get(language, language[:2] if len(language) >= 2 else "en")
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._mlx_repo,
            language=lang_code,
            verbose=False,
        )
        return (result.get("text") or "").strip()

    def unload(self) -> None:
        self._loaded = False
        super().unload()


# ─── Qwen3-ASR via mlx-qwen3-asr (Apple Silicon) ───────────────────────────────


_MLX_QWEN3_ASR_REPO_BY_HF: dict[str, str] = {
    "Qwen/Qwen3-ASR-0.6B": "mlx-community/Qwen3-ASR-0.6B-bf16",
    "Qwen/Qwen3-ASR-1.7B": "mlx-community/Qwen3-ASR-1.7B-bf16",
}


class MLXQwen3ASRModel(ASRModel):
    """Qwen3-ASR via mlx-qwen3-asr (Apple Silicon)."""

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._mlx_repo = _MLX_QWEN3_ASR_REPO_BY_HF.get(spec.hf_id, spec.hf_id)
        self._model = None

    def load(self) -> None:
        import mlx_qwen3_asr  # type: ignore

        # The mlx-qwen3-asr API has shifted; try the most common entry points.
        try:
            self._model = mlx_qwen3_asr.load(self._mlx_repo)
        except AttributeError:
            try:
                self._model = mlx_qwen3_asr.from_pretrained(self._mlx_repo)
            except AttributeError:
                # Fall back to a Model class
                cls = getattr(mlx_qwen3_asr, "Model", None) or getattr(
                    mlx_qwen3_asr, "Qwen3ASR", None
                )
                if cls is None:
                    raise RuntimeError(
                        "mlx-qwen3-asr API unrecognized — "
                        "neither load(), from_pretrained(), Model, nor Qwen3ASR found"
                    )
                self._model = cls.from_pretrained(self._mlx_repo)

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        # mlx-qwen3-asr accepts a numpy float32 16k array OR a file path.
        # Try the most common method names.
        for fn_name in ("transcribe", "generate", "__call__"):
            fn = getattr(self._model, fn_name, None)
            if not fn:
                continue
            try:
                out = fn(audio, sample_rate=sample_rate)
            except TypeError:
                try:
                    out = fn(audio)
                except Exception:
                    continue
            if isinstance(out, dict):
                return (out.get("text") or "").strip()
            if isinstance(out, str):
                return out.strip()
            if hasattr(out, "text"):
                return str(out.text).strip()
        raise RuntimeError("mlx-qwen3-asr model has no recognized transcribe method")

    def unload(self) -> None:
        self._model = None
        super().unload()


# ─── Whisper via faster-whisper (CTranslate2) ──────────────────────────────────


class FasterWhisperModel(ASRModel):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model = None

    def load(self) -> None:
        from faster_whisper import WhisperModel  # type: ignore

        compute_type = "float16"
        try:
            import torch  # type: ignore

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
            compute_type = "int8"

        self._model = WhisperModel(
            self.spec.hf_id,
            device=device,
            compute_type=compute_type,
        )

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        # faster-whisper accepts a numpy float32 array directly. It expects
        # 2-letter ISO language codes ("en", "es", …).
        lang_code = {
            "english": "en",
            "spanish": "es",
            "french": "fr",
            "german": "de",
            "lithuanian": "lt",
            "portuguese": "pt",
            "italian": "it",
            "dutch": "nl",
            "polish": "pl",
        }.get(language, language[:2] if len(language) >= 2 else "en")
        bsz = self.spec.beam_size if self.spec.beam_size > 0 else 5
        segments, _info = self._model.transcribe(
            audio,
            language=lang_code,
            beam_size=bsz,
            vad_filter=False,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def unload(self) -> None:
        self._model = None
        super().unload()


# ─── Parakeet via NeMo ─────────────────────────────────────────────────────────


class NeMoParakeetModel(ASRModel):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr  # type: ignore

        # NeMo selects the right ModelPT subclass automatically.
        self._model = nemo_asr.models.ASRModel.from_pretrained(self.spec.hf_id)
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")
            self._model.eval()
        except Exception:
            pass

        # Apply beam+LM decoding if configured in the ModelSpec.
        if self.spec.beam_size > 0:
            self._apply_beam_lm()

    def _apply_beam_lm(self) -> None:
        """Switch from greedy to beam search + n-gram LM fusion."""
        import copy
        from omegaconf import open_dict  # type: ignore

        # Resolve the LM file path: try local first, then download from HF.
        lm_path = self.spec.lm_local_path
        if not lm_path or not os.path.exists(lm_path):
            if self.spec.lm_hf_filename:
                try:
                    from huggingface_hub import hf_hub_download  # type: ignore

                    lm_path = hf_hub_download(
                        self.spec.hf_id,
                        self.spec.lm_hf_filename,
                    )
                    print(f"  downloaded LM: {self.spec.lm_hf_filename} → {lm_path}")
                except Exception as e:
                    print(f"  ! LM download failed: {e} — falling back to greedy")
                    return
            else:
                print(f"  ! no LM path configured — staying greedy")
                return

        new_cfg = copy.deepcopy(self._model.cfg.decoding)
        with open_dict(new_cfg):
            new_cfg.strategy = "maes"  # TDT beam search
            new_cfg.beam.beam_size = self.spec.beam_size
            new_cfg.beam.return_best_hypothesis = True
            new_cfg.beam.ngram_lm_model = str(lm_path)
            new_cfg.beam.ngram_lm_alpha = self.spec.lm_alpha

        self._model.change_decoding_strategy(new_cfg)
        print(
            f"  beam+LM: beam={self.spec.beam_size} alpha={self.spec.lm_alpha} "
            f"lm={os.path.basename(str(lm_path))}"
        )

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        import soundfile as sf  # type: ignore

        # NeMo's transcribe() takes a list of file paths.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sample_rate, format="WAV")
            tmp_path = tmp.name
        try:
            outs = self._model.transcribe([tmp_path], batch_size=1, verbose=False)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        # NeMo's API has shifted between versions. Normalize to a string.
        item = outs[0] if outs else ""
        if isinstance(item, list):
            item = item[0] if item else ""
        if hasattr(item, "text"):
            item = item.text
        if isinstance(item, tuple):
            item = item[0]
        return str(item).strip()

    def unload(self) -> None:
        self._model = None
        super().unload()


# ─── Qwen3-ASR via HF transformers ─────────────────────────────────────────────


class Qwen3ASRModel(ASRModel):
    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModel, AutoProcessor  # type: ignore

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        # Qwen3-ASR has a custom model class — must use trust_remote_code=True
        # so transformers loads the architecture from the repo's own code.
        self._processor = AutoProcessor.from_pretrained(
            self.spec.hf_id, trust_remote_code=True
        )
        self._model = AutoModel.from_pretrained(
            self.spec.hf_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self._model.eval()

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        import torch  # type: ignore

        lang_label = language.capitalize() if language else "English"
        prompt = f"Transcribe the audio in {lang_label}. Output only the transcription."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(
            text=text,
            audios=[audio],
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with torch.inference_mode():
            out_ids = self._model.generate(**inputs, max_new_tokens=440)
        out = self._processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        return out.strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        super().unload()


# ─── Gemma 4 (multimodal LLM with audio) via HF transformers ───────────────────


class Gemma4AudioModel(ASRModel):
    """Gemma 4 E2B/E4B IT — audio-capable multimodal LLM. Prompt for transcription."""

    PROMPT_TEMPLATE = (
        "Transcribe the audio{lang_hint} exactly as spoken. "
        "Output only the transcription, no preamble or commentary."
    )
    # Smaller chunks keep per-forward activation memory under ~4 GB on T4
    # (16 GB VRAM). At 28s the E4B variant OOMs on longer-than-average clips
    # (observed on Lithuanian FLEURS + CV22): the model weights are ~10 GB and
    # each forward needs ~5 GB of activations, which exceeds T4 capacity.
    CHUNK_SECONDS = 16.0
    OVERLAP_SECONDS = 0.5

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._model = None
        self._processor = None

    def load(self) -> None:
        import torch  # type: ignore
        from transformers import AutoProcessor  # type: ignore

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Pick the right model class. transformers 5.5+ ships
        # Gemma4ForConditionalGeneration (actual architecture per the HF
        # config.json); older versions may only have Gemma3nForConditionalGeneration
        # or the AutoModelForImageTextToText fallback.
        ModelCls = None
        for name in (
            "Gemma4ForConditionalGeneration",
            "Gemma3nForConditionalGeneration",
            "AutoModelForImageTextToText",
            "AutoModelForCausalLM",
        ):
            try:
                module = __import__("transformers", fromlist=[name])
                ModelCls = getattr(module, name)
                break
            except (ImportError, AttributeError):
                continue
        if ModelCls is None:
            raise RuntimeError("no usable Gemma 4 model class found in transformers")

        # trust_remote_code lets the model repo provide its own processor /
        # model code, which is sometimes how Gemma 4 multimodal ships.
        self._processor = AutoProcessor.from_pretrained(
            self.spec.hf_id, trust_remote_code=True
        )
        self._model = ModelCls.from_pretrained(
            self.spec.hf_id,
            torch_dtype=dtype,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.eval()

    def _transcribe_chunk(self, audio_chunk, sample_rate: int, language: str = "english") -> str:
        import torch  # type: ignore

        lang_hint = f" in {language.capitalize()}" if language and language != "english" else ""
        prompt = self.PROMPT_TEMPLATE.format(lang_hint=lang_hint)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_chunk},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            k: (v.to(self._model.device) if hasattr(v, "to") else v)
            for k, v in inputs.items()
        }
        with torch.inference_mode():
            out_ids = self._model.generate(
                **inputs, max_new_tokens=440, do_sample=False
            )
        gen = out_ids[:, inputs["input_ids"].shape[1] :]
        text = self._processor.batch_decode(gen, skip_special_tokens=True)[0]
        # Free per-chunk activation memory so successive chunks don't pile up
        # on the 16 GB T4 VRAM.
        del inputs, out_ids, gen
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return text.strip()

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        chunk_len = int(self.CHUNK_SECONDS * sample_rate)
        if len(audio) <= chunk_len:
            return self._transcribe_chunk(audio, sample_rate, language)

        step = chunk_len - int(self.OVERLAP_SECONDS * sample_rate)
        parts: list[str] = []
        for start in range(0, len(audio), step):
            chunk = audio[start : start + chunk_len]
            if len(chunk) < int(0.2 * sample_rate):
                break
            parts.append(self._transcribe_chunk(chunk, sample_rate, language))
        return " ".join(p for p in parts if p).strip()

    def unload(self) -> None:
        self._model = None
        self._processor = None
        super().unload()


# ─── Gemma 4 with LoRA adapter (fine-tuned ASR) ────────────────────────────────


class Gemma4LoRAModel(Gemma4AudioModel):
    """Gemma 4 + PEFT LoRA adapter for fine-tuned ASR.

    Loads the base Gemma 4 model via the parent class, then applies a
    LoRA adapter from ``spec.adapter_id`` (HF repo or local path). The
    adapter is small (~140 MB) while the base model is ~8 GB, so the
    adapter can live on HuggingFace separately from the base weights.

    Usage in speechbench:
        speechbench run --models gemma-4-E4B-it-lt-asr --datasets common_voice_25_lt
    """

    def load(self) -> None:
        super().load()
        if not self.spec.adapter_id:
            return
        from peft import PeftModel  # type: ignore

        self._model = PeftModel.from_pretrained(
            self._model, self.spec.adapter_id
        )
        # PeftModel wraps the base model; re-set to inference mode.
        self._model.train(False)


# ─── DashScope Qwen3.5-Omni API ────────────────────────────────────────────────


class DashScopeOmniModel(ASRModel):
    """Qwen3.5-Omni via Alibaba DashScope OpenAI-compatible API.

    Audio is sent inline (base64 wav). Each call counts against the user's
    DashScope quota — billing is separate from GPU-VM costs.
    """

    PROMPT = "Transcribe the audio exactly as spoken. Output only the transcription, no preamble."

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._client = None
        self._api_model_name = spec.hf_id  # e.g. 'qwen3.5-omni-flash'

    def load(self) -> None:
        from openai import OpenAI  # type: ignore

        api_key = os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("DASHSCOPE_KEY")
        if not api_key:
            raise RuntimeError(
                "DashScope API key not set. Pass --api-keys DASHSCOPE_API_KEY=… "
                "to speechbench launch (or export DASHSCOPE_API_KEY locally)."
            )
        base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        import base64
        import io

        import soundfile as sf  # type: ignore

        buf = io.BytesIO()
        sf.write(buf, audio, sample_rate, format="WAV")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode("ascii")
        data_url = f"data:audio/wav;base64,{b64}"

        resp = self._client.chat.completions.create(
            model=self._api_model_name,
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_audio", "input_audio": {"data": data_url, "format": "wav"}},
                        {"type": "text", "text": self.PROMPT},
                    ],
                }
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    def unload(self) -> None:
        self._client = None


# ─── Registry ──────────────────────────────────────────────────────────────────


def _whisper(key: str, hf_id: str, vram: int, sec_per_s: float, load_s: float) -> ModelSpec:
    return ModelSpec(
        key=key,
        family="whisper",
        backend="transformers",
        hf_id=hf_id,
        min_vram_gb=vram,
        sec_per_audio_sec=sec_per_s,
        load_seconds=load_s,
        description=f"OpenAI Whisper — {hf_id}",
    )


def _fw(key: str, ct2_id: str, vram: int, sec_per_s: float, load_s: float) -> ModelSpec:
    return ModelSpec(
        key=key,
        family="whisper",
        backend="faster-whisper",
        hf_id=ct2_id,
        min_vram_gb=vram,
        sec_per_audio_sec=sec_per_s,
        load_seconds=load_s,
        description=f"faster-whisper (CTranslate2) — {ct2_id}",
    )


def _parakeet(key: str, hf_id: str, vram: int, sec_per_s: float, load_s: float,
              beam_size: int = 0, lm_alpha: float = 0.5,
              lm_hf_filename: str = "", lm_local_path: str = "",
              description: str = "") -> ModelSpec:
    if not description:
        desc = f"NVIDIA NeMo Parakeet — {hf_id}"
        if beam_size > 0:
            lm_name = lm_hf_filename.replace(".arpa", "") if lm_hf_filename else "LM"
            desc += f" + beam={beam_size} + {lm_name} α={lm_alpha}"
    else:
        desc = description
    return ModelSpec(
        key=key,
        family="parakeet",
        backend="nemo",
        hf_id=hf_id,
        min_vram_gb=vram,
        sec_per_audio_sec=sec_per_s,
        load_seconds=load_s,
        description=desc,
        beam_size=beam_size,
        lm_alpha=lm_alpha,
        lm_hf_filename=lm_hf_filename,
        lm_local_path=lm_local_path,
    )


MODELS: dict[str, ModelSpec] = {}


def _register(specs: list[ModelSpec]) -> None:
    for s in specs:
        MODELS[s.key] = s


_register(
    [
        # Whisper — HF transformers (English-only variants)
        _whisper("whisper-tiny.en", "openai/whisper-tiny.en", 2, 0.020, 25),
        _whisper("whisper-base.en", "openai/whisper-base.en", 2, 0.025, 25),
        _whisper("whisper-small.en", "openai/whisper-small.en", 3, 0.040, 30),
        _whisper("whisper-medium.en", "openai/whisper-medium.en", 5, 0.070, 40),
        # Whisper — HF transformers (multilingual variants — needed for Spanish etc.)
        _whisper("whisper-tiny", "openai/whisper-tiny", 2, 0.020, 25),
        _whisper("whisper-base", "openai/whisper-base", 2, 0.025, 25),
        _whisper("whisper-small", "openai/whisper-small", 3, 0.040, 30),
        _whisper("whisper-medium", "openai/whisper-medium", 5, 0.070, 40),
        _whisper("whisper-large-v2", "openai/whisper-large-v2", 8, 0.110, 60),
        _whisper("whisper-large-v3", "openai/whisper-large-v3", 8, 0.110, 60),
        _whisper("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", 8, 0.050, 60),
        _whisper("distil-large-v3", "distil-whisper/distil-large-v3", 6, 0.030, 45),
        # Whisper — faster-whisper / CTranslate2
        _fw("fw-large-v3", "large-v3", 5, 0.030, 30),
        _fw("fw-large-v3-turbo", "large-v3-turbo", 4, 0.018, 30),
        _fw("fw-distil-large-v3", "distil-large-v3", 4, 0.012, 30),
        # Parakeet — NeMo
        _parakeet("parakeet-tdt-0.6b-v3", "nvidia/parakeet-tdt-0.6b-v3", 4, 0.010, 50),
        _parakeet("parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v2", 4, 0.010, 50),
        _parakeet("parakeet-tdt-1.1b", "nvidia/parakeet-tdt-1.1b", 6, 0.018, 70),
        _parakeet("parakeet-rnnt-1.1b", "nvidia/parakeet-rnnt-1.1b", 6, 0.022, 70),
        _parakeet("parakeet-rnnt-0.6b", "nvidia/parakeet-rnnt-0.6b", 4, 0.012, 50),
        _parakeet("parakeet-ctc-1.1b", "nvidia/parakeet-ctc-1.1b", 5, 0.008, 60),
        _parakeet("parakeet-ctc-0.6b", "nvidia/parakeet-ctc-0.6b", 4, 0.006, 50),
        _parakeet("parakeet-tdt_ctc-110m", "nvidia/parakeet-tdt_ctc-110m", 2, 0.005, 30),
        # Whisper beam=5 variants (fair comparison with beam+LM parakeet)
        ModelSpec(key="whisper-large-v3+beam5", family="whisper", backend="transformers",
                  hf_id="openai/whisper-large-v3", min_vram_gb=8,
                  sec_per_audio_sec=0.15, load_seconds=60, beam_size=5,
                  description="Whisper large-v3 + beam=5"),
        ModelSpec(key="whisper-large-v3-turbo+beam5", family="whisper", backend="transformers",
                  hf_id="openai/whisper-large-v3-turbo", min_vram_gb=8,
                  sec_per_audio_sec=0.07, load_seconds=60, beam_size=5,
                  description="Whisper large-v3-turbo + beam=5"),
        # Fine-tunes of parakeet-tdt-0.6b-v3 on specific languages
        _parakeet("parakeet-tdt-lt", "sliderforthewin/parakeet-tdt-lt", 4, 0.010, 50),
        # Beam+LM variants — same model, different decoding strategies
        _parakeet("parakeet-tdt-lt+domain5gram",
                  "sliderforthewin/parakeet-tdt-lt", 4, 0.040, 50,
                  beam_size=4, lm_alpha=0.5, lm_hf_filename="lt_domain_5gram.arpa"),
        _parakeet("parakeet-tdt-lt+europarl5gram",
                  "sliderforthewin/parakeet-tdt-lt", 4, 0.040, 50,
                  beam_size=4, lm_alpha=0.5, lm_hf_filename="lt_europarl_wiki_subs_5gram.arpa"),
        # Qwen3-ASR
        ModelSpec(
            key="qwen3-asr-0.6b",
            family="qwen3-asr",
            backend="transformers",
            hf_id="Qwen/Qwen3-ASR-0.6B",
            min_vram_gb=4,
            sec_per_audio_sec=0.08,
            load_seconds=45,
            description="Qwen3-ASR-0.6B",
        ),
        ModelSpec(
            key="qwen3-asr-1.7b",
            family="qwen3-asr",
            backend="transformers",
            hf_id="Qwen/Qwen3-ASR-1.7B",
            min_vram_gb=6,
            sec_per_audio_sec=0.15,
            load_seconds=60,
            description="Qwen3-ASR-1.7B",
        ),
        # Qwen3.5-Omni — DashScope API
        ModelSpec(
            key="qwen3.5-omni-plus",
            family="qwen3.5-omni",
            backend="dashscope-api",
            hf_id="qwen3.5-omni-plus",
            min_vram_gb=0,
            sec_per_audio_sec=0.30,
            load_seconds=2,
            needs_dashscope_key=True,
            is_api=True,
            description="Qwen3.5-Omni Plus (DashScope API)",
        ),
        ModelSpec(
            key="qwen3.5-omni-flash",
            family="qwen3.5-omni",
            backend="dashscope-api",
            hf_id="qwen3.5-omni-flash",
            min_vram_gb=0,
            sec_per_audio_sec=0.20,
            load_seconds=2,
            needs_dashscope_key=True,
            is_api=True,
            description="Qwen3.5-Omni Flash (DashScope API)",
        ),
        ModelSpec(
            key="qwen3.5-omni-light",
            family="qwen3.5-omni",
            backend="dashscope-api",
            hf_id="qwen3.5-omni-light",
            min_vram_gb=0,
            sec_per_audio_sec=0.15,
            load_seconds=2,
            needs_dashscope_key=True,
            is_api=True,
            description="Qwen3.5-Omni Light (DashScope API)",
        ),
        # Gemma 4 (audio-capable multimodal LLM)
        ModelSpec(
            key="gemma-4-E4B-it",
            family="gemma4",
            backend="transformers",
            hf_id="google/gemma-4-E4B-it",
            min_vram_gb=12,
            sec_per_audio_sec=0.40,
            load_seconds=90,
            description="Gemma 4 E4B-IT — multimodal LLM, prompt-based ASR",
        ),
        ModelSpec(
            key="gemma-4-E2B-it",
            family="gemma4",
            backend="transformers",
            hf_id="google/gemma-4-E2B-it",
            min_vram_gb=8,
            sec_per_audio_sec=0.30,
            load_seconds=70,
            description="Gemma 4 E2B-IT — multimodal LLM, prompt-based ASR",
        ),
        # Gemma 4 + LoRA fine-tuned for Lithuanian ASR
        ModelSpec(
            key="gemma-4-E4B-it-lt-asr",
            family="gemma4",
            backend="transformers",
            hf_id="google/gemma-4-E4B-it",
            adapter_id="sliderforthewin/gemma-4-E4B-it-lt-asr",
            min_vram_gb=12,
            sec_per_audio_sec=0.40,
            load_seconds=100,
            description="Gemma 4 E4B-IT + Lithuanian ASR LoRA (WER 29.74% on CV25 LT)",
        ),
    ]
)


# ─── Construction (used at runtime on the VM) ──────────────────────────────────


class SkippedModel(ASRModel):
    """Placeholder for a model that can't run on the current platform."""

    def __init__(self, spec: ModelSpec, reason: str):
        super().__init__(spec)
        self._reason = reason

    def load(self) -> None:
        raise RuntimeError(f"skipped: {self._reason}")

    def transcribe(self, audio, sample_rate: int = 16000, language: str = "english") -> str:
        raise RuntimeError(f"skipped: {self._reason}")


def make_model(spec: ModelSpec) -> ASRModel:
    """Construct the right wrapper for the current platform.

    On Apple Silicon (Mac arm64) we prefer:
      - mlx-whisper for Whisper (when installed)
      - mlx-qwen3-asr for Qwen3-ASR (when installed)
      - NeMo directly for Parakeet (works on Mac too — same as ScribeBox)
    Otherwise (Linux+CUDA, the GCP DLVM path) we use the standard HF /
    faster-whisper / NeMo wrappers.

    `faster-whisper` (CTranslate2) is intentionally skipped on Mac because the
    same weights are also covered by mlx-whisper, which is a more native fit.
    """
    # Whisper
    if spec.family == "whisper":
        if spec.backend == "faster-whisper":
            if is_mac_arm():
                return SkippedModel(
                    spec,
                    "faster-whisper not run on darwin/arm64 — same weights are "
                    "covered by the mlx-whisper variants",
                )
            return FasterWhisperModel(spec)
        # transformers backend
        if is_mac_arm() and have_mlx_whisper():
            return MLXWhisperModel(spec)
        return HFWhisperModel(spec)

    # Parakeet — NeMo on both Linux/CUDA and Mac (CPU)
    if spec.backend == "nemo":
        return NeMoParakeetModel(spec)

    # Qwen3-ASR
    if spec.family == "qwen3-asr":
        if is_mac_arm() and have_mlx_qwen3_asr():
            return MLXQwen3ASRModel(spec)
        return Qwen3ASRModel(spec)

    # Gemma 4 — HF Transformers everywhere; on Mac it falls back to MPS/CPU
    if spec.family == "gemma4":
        if spec.adapter_id:
            return Gemma4LoRAModel(spec)
        return Gemma4AudioModel(spec)

    # DashScope API models — work anywhere
    if spec.backend == "dashscope-api":
        return DashScopeOmniModel(spec)

    raise ValueError(f"Unknown backend/family for {spec.key}: {spec.backend}/{spec.family}")


def resolve(patterns: list[str]) -> list[str]:
    """Resolve model glob patterns against MODELS keys."""
    import fnmatch

    out: list[str] = []
    seen: set[str] = set()
    for p in patterns:
        matches = [k for k in MODELS if fnmatch.fnmatchcase(k, p)]
        if not matches:
            raise ValueError(f"No model matches pattern: {p!r}")
        for m in matches:
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def list_model_keys() -> list[str]:
    return list(MODELS.keys())
