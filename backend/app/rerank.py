"""Rerank the search candidates with a cross-encoder (bge-reranker-v2-m3).

What: `Reranker.score(question, passages)` returns one relevance score per
passage, between 0 and 1 (higher = more relevant).

Why: the search compares a question vector with chunk vectors that were
computed separately, without seeing the question ("bi-encoder"). That is
fast enough for the whole collection, but approximate. A cross-encoder reads
the question and one passage together, as a single input, so every word of
the question is compared with every word of the passage. It is much more
accurate, but needs one model run per (question, passage) pair: too slow
for the whole collection, fine for the 30 candidates of the search (about
1 s on the GPU, 50 s on the CPU: docs/vram-budget.md).

How: FlagEmbedding's FlagReranker. The model outputs a raw score (a logit);
normalize=True passes it through a sigmoid, so scores are between 0 and 1.
The passage text includes the "Title > Section" header when the setting
chunk_add_header is on (the same text the embedder saw; see app.pipeline).
"""

from FlagEmbedding import FlagReranker

from app.model_files import local_model_path


class Reranker:
    def __init__(self, model_name: str, device: str, fp16: bool, batch_size: int, max_length: int):
        self.batch_size = batch_size
        self.max_length = max_length
        self.model = FlagReranker(
            local_model_path(model_name),
            use_fp16=fp16 and device == "cuda",  # half precision only on the GPU
            devices=device,
        )

    def score(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self.model.compute_score(
            [(question, passage) for passage in passages],
            batch_size=self.batch_size, max_length=self.max_length, normalize=True,
        )
        if isinstance(scores, float):  # FlagReranker returns a bare number for a single pair
            scores = [scores]
        return [float(s) for s in scores]
