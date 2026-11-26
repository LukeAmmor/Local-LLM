"""Search Qdrant for the chunks that best match a question, with access control.

What: `search()` runs in one of three modes and returns `Candidate`s (the
chunk's payload, its rank in each search, and a score):
  dense   meaning-based: cosine similarity between the dense vectors.
          Works across languages (a French question finds English chunks).
  sparse  keyword-like: dot product between the sparse vectors (bge-m3's
          learned word weights). Exact terms, names, numbers... but a
          French word does not match its English translation.
  hybrid  both searches, merged with Reciprocal Rank Fusion (RRF).

Access control: `access_filter()` ("allowed_groups contains one of the
user's groups") is part of EVERY Qdrant query. Qdrant only ranks chunks the
user may read, so a forbidden chunk is never returned, never fused and never
given to the LLM. `check_access()` verifies the results again in Python: a
tripwire that should never fire. The public group is always added to the
user's groups (`user_groups()`): public documents are for everyone.

RRF (Cormack et al., 2009): a chunk's fused score is the sum, over the
searches that found it, of 1 / (k + rank), where rank 1 is the best. Only
ranks are used, not scores, so the two searches do not need comparable
scores (cosine similarity and sparse dot products have different scales).
The constant k (60, the value from the paper) keeps the first ranks from
dominating: rank 1 gives 1/61, rank 2 gives 1/62. A chunk found by both
searches usually beats a chunk found by one search only.

Why our own RRF rather than Qdrant's built-in fusion: it is a few lines, and
it keeps each chunk's dense and sparse rank, which `app.ask --verbose`
shows for the experiments.
"""

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models

from app.embeddings import Embedding
from app.vectorstore import DENSE, SPARSE


@dataclass
class Candidate:
    payload: dict  # what Qdrant stores with the chunk: text, title, pages, allowed_groups...
    ranks: dict[str, int] = field(default_factory=dict)  # {"dense": 3, "sparse": 12}; 1 = best
    score: float = 0.0  # cosine (dense), dot product (sparse) or RRF score (hybrid)
    rerank_score: float | None = None  # set by the reranker, between 0 and 1
    cited: bool = False  # set after generation: does the answer cite this passage?


def user_groups(groups: list[str], public_group: str) -> list[str]:
    """The groups used for the access filter: the user's groups plus the public group."""
    return sorted(set(groups) | {public_group})


def access_filter(groups: list[str]) -> models.Filter:
    """Qdrant filter: the chunk's allowed_groups contains at least one of these groups."""
    return models.Filter(must=[models.FieldCondition(key="allowed_groups", match=models.MatchAny(any=groups))])


def rrf_score(ranks: list[int], k: int) -> float:
    """Reciprocal Rank Fusion: the sum of 1 / (k + rank) over the searches that found the chunk."""
    return sum(1.0 / (k + rank) for rank in ranks)


def query(client: QdrantClient, collection: str, using: str, vector, groups: list[str],
          limit: int) -> list[models.ScoredPoint]:
    """One search on one vector ("dense" or "sparse"), with the access filter inside the query."""
    return client.query_points(
        collection, query=vector, using=using, query_filter=access_filter(groups),
        limit=limit, with_payload=True,
    ).points


def search(client: QdrantClient, collection: str, embedding: Embedding, groups: list[str],
           mode: str, top_k: int, rrf_k: int) -> list[Candidate]:
    """The top_k chunks for the question embedding, best first (only chunks `groups` may read)."""
    results = {}  # search name -> points, best first
    if mode in ("dense", "hybrid"):
        results[DENSE] = query(client, collection, DENSE, embedding.dense, groups, top_k)
    if mode in ("sparse", "hybrid"):
        sparse = models.SparseVector(indices=embedding.sparse.indices, values=embedding.sparse.values)
        results[SPARSE] = query(client, collection, SPARSE, sparse, groups, top_k)

    candidates: dict[str, Candidate] = {}  # chunk_id -> Candidate (a chunk found twice is one candidate)
    for name, points in results.items():
        for rank, point in enumerate(points, start=1):
            candidate = candidates.setdefault(point.payload["chunk_id"], Candidate(payload=point.payload))
            candidate.ranks[name] = rank
            candidate.score = point.score  # one search: keep its own score
    if mode == "hybrid":
        for candidate in candidates.values():
            candidate.score = rrf_score(list(candidate.ranks.values()), rrf_k)

    ranked = sorted(candidates.values(), key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


def check_access(candidates: list[Candidate], groups: list[str]) -> None:
    """Raise if a candidate is not readable by `groups` (the Qdrant filter should make this impossible)."""
    for c in candidates:
        if not set(c.payload["allowed_groups"]) & set(groups):
            raise PermissionError(f"access filter failed: chunk {c.payload['chunk_id']} "
                                  f"(allowed {c.payload['allowed_groups']}) returned for groups {groups}")
