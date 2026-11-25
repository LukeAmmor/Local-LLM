"""Resolve a model name to a local folder, downloading only the files we need.

What: `local_model_path("BAAI/bge-m3")` downloads the model from Hugging Face
once (into the usual Hugging Face cache) and returns the local folder. If the
name is already a local folder (for example our fine-tuned model in models/),
it is returned unchanged.

Why: FlagEmbedding downloads the WHOLE repository when given a name. The
bge-m3 repository also contains an ONNX copy of the weights (2.1 GiB) that we
never use. Passing a local folder to FlagEmbedding skips its own download.
"""

from pathlib import Path

from huggingface_hub import snapshot_download

# Files we never need: ONNX export and other frameworks' weights.
IGNORE_PATTERNS = ["onnx/*", "*.onnx", "*.onnx_data", "flax_model.msgpack", "rust_model.ot", "tf_model.h5"]


def local_model_path(name_or_path: str) -> str:
    """Return a local folder containing the model (download it the first time)."""
    if Path(name_or_path).exists():
        return name_or_path
    return snapshot_download(repo_id=name_or_path, ignore_patterns=IGNORE_PATTERNS)
