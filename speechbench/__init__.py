"""speechbench — ASR benchmark suite for GCP spot GPU instances."""

# Tell HuggingFace transformers to NOT auto-import tensorflow / flax. We only
# need PyTorch. On macOS systems that have tensorflow-macos installed, the TF
# Metal plugin can fatally abort the process during transformers import (it
# tries to register a Metal platform that another tf process already
# registered). Set these *before* any transformers import — at package load.
import os as _os
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("USE_FLAX", "0")
_os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# huggingface_hub 1.x introduced an "Xet" download protocol via hf-xet that,
# as of 1.4.3, can hang indefinitely finalizing large-file downloads on
# macOS. Force the classic HTTP download path until it stabilizes.
_os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

__version__ = "0.1.0"
