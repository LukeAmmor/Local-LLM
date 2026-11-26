"""Tests for app.retrieval: RRF fusion and access control.

The first tests use a fake Qdrant client (fixed results, records the
filters it receives). The last one uses a temporary collection in the real
Qdrant (skipped if Qdrant is not running) to check that the filter really
hides the chunks of other groups, in every search mode.
"""

import uuid
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.embeddings import Embedding, SparseVector
from app.retrieval import Candidate, check_access, rrf_score, search, user_groups
from app.vectorstore import ensure_collection, upsert_chunks

EMBEDDING = Embedding(dense=[1.0] * 1024, sparse=SparseVector(indices=[1, 2], values=[1.0, 0.5]))


def point(chunk_id: str, score: float, groups=("hr",)) -> models.ScoredPoint:
    return models.ScoredPoint(id=str(uuid.uuid4()), version=0, score=score,
                              payload={"chunk_id": chunk_id, "allowed_groups": list(groups)})


class FakeQdrant:
    """Returns fixed results per vector name and records every filter it receives."""

    def __init__(self, results: dict[str, list[models.ScoredPoint]]):
        self.results = results
        self.filters = []

    def query_points(self, collection, query, using, query_filter, limit, with_payload):
        self.filters.append(query_filter)
        return SimpleNamespace(points=self.results[using][:limit])


def test_user_groups_always_include_public():
    assert user_groups(["hr"], "public") == ["hr", "public"]
    assert user_groups([], "public") == ["public"]


def test_rrf_found_by_both_searches_beats_first_in_one():
    assert rrf_score([2, 2], k=60) > rrf_score([1], k=60)  # 2/62 > 1/61
    assert rrf_score([1], k=60) == pytest.approx(1 / 61)


def test_hybrid_search_fuses_the_two_rankings():
    fake = FakeQdrant({
        "dense": [point("A", 0.9), point("B", 0.8), point("C", 0.7)],
        "sparse": [point("B", 12.0), point("D", 8.0)],
    })
    ranked = search(fake, "kb", EMBEDDING, ["hr", "public"], "hybrid", top_k=10, rrf_k=60)
    # B: 1/62 + 1/61, A: 1/61, D: 1/62, C: 1/63
    assert [c.payload["chunk_id"] for c in ranked] == ["B", "A", "D", "C"]
    assert ranked[0].ranks == {"dense": 2, "sparse": 1}
    assert ranked[2].ranks == {"sparse": 2}  # D was not found by the dense search


def test_single_mode_keeps_its_own_scores_and_top_k():
    fake = FakeQdrant({"dense": [point("A", 0.9), point("B", 0.8), point("C", 0.7)]})
    ranked = search(fake, "kb", EMBEDDING, ["hr"], "dense", top_k=2, rrf_k=60)
    assert [(c.payload["chunk_id"], c.score) for c in ranked] == [("A", 0.9), ("B", 0.8)]


def test_every_query_carries_the_access_filter():
    fake = FakeQdrant({"dense": [point("A", 0.9)], "sparse": [point("A", 3.0)]})
    search(fake, "kb", EMBEDDING, ["hr", "public"], "hybrid", top_k=10, rrf_k=60)
    assert len(fake.filters) == 2  # one dense query, one sparse query
    for f in fake.filters:
        condition = f.must[0]
        assert condition.key == "allowed_groups" and condition.match.any == ["hr", "public"]


def test_check_access_raises_on_a_forbidden_chunk():
    check_access([Candidate(payload={"chunk_id": "ok", "allowed_groups": ["hr"]})], ["hr", "public"])
    with pytest.raises(PermissionError):
        check_access([Candidate(payload={"chunk_id": "x", "allowed_groups": ["procurement"]})], ["hr", "public"])


@pytest.fixture
def qdrant_collection():
    """A temporary collection with 3 identical chunks of different groups (deleted afterwards)."""
    client = QdrantClient(url=get_settings().qdrant_url, timeout=5)
    try:
        client.get_collections()
    except Exception:  # noqa: BLE001
        pytest.skip("Qdrant is not running")
    name = f"test_access_{uuid.uuid4().hex[:8]}"
    ensure_collection(client, name)
    payloads = [{"chunk_id": f"{g}-doc:0", "doc_id": f"{g}-doc", "allowed_groups": [g]}
                for g in ("hr", "procurement", "public")]
    upsert_chunks(client, name, payloads, [EMBEDDING] * 3)
    yield client, name
    client.delete_collection(name)


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_real_qdrant_hides_other_groups(qdrant_collection, mode):
    client, name = qdrant_collection

    def doc_ids(groups):
        return sorted(c.payload["doc_id"] for c in search(client, name, EMBEDDING, groups, mode, 10, 60))

    # The 3 chunks have the same vectors: only the filter decides what comes back.
    assert doc_ids(user_groups(["hr"], "public")) == ["hr-doc", "public-doc"]
    assert doc_ids(user_groups(["finance"], "public")) == ["public-doc"]  # a group without documents
    assert doc_ids(["nobody"]) == []
