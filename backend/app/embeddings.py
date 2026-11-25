"""bge-m3 embeddings: one dense vector and one sparse vector per text.

What: `Embedder.encode(texts)` returns, for each text,
  - dense:  1024 numbers describing the meaning of the text (normalised, so
            cosine similarity = dot product). Works across languages: a French
            question and an English passage with the same meaning are close.
  - sparse: a few {token id: weight} pairs, the important words of the text
            and how important they are. Close to keyword search (like BM25),
            but the weights are learned by the model.
Qdrant stores both, and hybrid search combines them (Phase 2).

Why dense and sparse are separate outputs: in Phase 6 we fine-tune only the
dense part. The sparse vectors must then still come from the base model,
because FlagEmbedding silently creates a RANDOM sparse head for a model folder
that has no `sparse_linear.pt`.

`load_tokenizer()` gives the bge-m3 tokenizer, used by chunking to count
tokens exactly like the embedder does.
"""

from dataclasses import dataclass

from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer

from app.model_files import local_model_path


@dataclass
class SparseVector:
    indices: list[int]  # token ids
    values: list[float]  # their weights


@dataclass
class Embedding:
    dense: list[float]
    sparse: SparseVector


def load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(local_model_path(model_name))


class Embedder:
    def __init__(self, model_name: str, device: str, fp16: bool, batch_size: int, max_length: int):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = BGEM3FlagModel(
            local_model_path(model_name),
            use_fp16=fp16 and device == "cuda",  # half precision only on the GPU
            devices=device,
        )

    def encode(self, texts: list[str]) -> list[Embedding]:
        out = self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        embeddings = []
        for dense, weights in zip(out["dense_vecs"], out["lexical_weights"]):
            # lexical_weights: {"token id as string": weight}
            sparse = SparseVector(indices=[int(t) for t in weights], values=[float(w) for w in weights.values()])
            embeddings.append(Embedding(dense=[float(x) for x in dense], sparse=sparse))
        return embeddings
