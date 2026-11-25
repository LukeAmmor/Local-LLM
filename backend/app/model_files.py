"""Resolve a model name to a local folder, downloading only the files we need.

What: `local_model_path("BAAI/bge-m3")` returns the local folder of the model.
  - If the name is already a local folder (for example our fine-tuned model
    in models/), it is returned unchanged.
  - If the model is already in the Hugging Face cache, the cached folder is
    used WITHOUT contacting the internet.
  - Otherwise the model is downloaded once from Hugging Face.

Why:
  - FlagEmbedding downloads the WHOLE repository when given a name. The
    bge-m3 repository also holds an ONNX copy of the weights (2.1 GiB) that
    we never use. Passing a local folder to FlagEmbedding skips its download.
  - A private, local assistant should not call an external service each time
    it starts. After the first download, everything works offline.
"""

from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

# Files we never need: ONNX export and other frameworks' weights.
IGNORE_PATTERNS = ["onnx/*", "*.onnx", "*.onnx_data", "flax_model.msgpack", "rust_model.ot", "tf_model.h5"]


def local_model_path(name_or_path: str) -> str:
    """Return a local folder containing the model (download it the first time only)."""
    if Path(name_or_path).exists():
        return name_or_path
    try:
        return snapshot_download(repo_id=name_or_path, ignore_patterns=IGNORE_PATTERNS, local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(repo_id=name_or_path, ignore_patterns=IGNORE_PATTERNS)
