"""Qdrant: create the collection, and store or delete the chunks of a document.

The collection holds one point per chunk:
  - vector "dense":  1024 numbers from bge-m3, compared with cosine similarity
  - vector "sparse": bge-m3 token weights (keyword-like matching)
  - payload: the chunk text and its metadata (doc_id, title, department,
    language, pages, section, allowed_groups...). The payload is what the
    search returns, and what the access filter checks.

Payload indexes on allowed_groups, doc_id, department and language make
filtering on these fields fast. The access-control filter of Phase 2 runs on
`allowed_groups` INSIDE the Qdrant query, so a chunk the user may not see is
never returned.

Point ids are derived from the chunk id ("<doc_id>:<index>") with UUID5, so
the same chunk always gets the same id. Re-indexing a document first deletes
its old points, then inserts the new ones: no duplicates, and no leftover
chunks if the new version has fewer chunks.
"""

import uuid

from qdrant_client import QdrantClient, models

from app.embeddings import Embedding

DENSE = "dense"
SPARSE = "sparse"
DENSE_SIZE = 1024  # bge-m3 output size
INDEXED_FIELDS = ["allowed_groups", "doc_id", "department", "language"]
UPSERT_BATCH = 64
POINT_NAMESPACE = uuid.UUID("6f1c2a52-6c2e-4a43-9d61-6d0c2b4b9a10")  # any fixed UUID works


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


def ensure_collection(client: QdrantClient, name: str) -> None:
    """Create the collection and its payload indexes if they do not exist yet."""
    if client.collection_exists(name):
        return
    client.create_collection(
        name,
        vectors_config={DENSE: models.VectorParams(size=DENSE_SIZE, distance=models.Distance.COSINE)},
        sparse_vectors_config={SPARSE: models.SparseVectorParams()},
    )
    for field in INDEXED_FIELDS:
        client.create_payload_index(name, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD)


def delete_document(client: QdrantClient, collection: str, doc_id: str) -> None:
    client.delete(
        collection,
        points_selector=models.FilterSelector(
            filter=models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        ),
        wait=True,
    )


def upsert_chunks(client: QdrantClient, collection: str, payloads: list[dict], embeddings: list[Embedding]) -> None:
    """Store chunks: payloads[i] (must contain "chunk_id") goes with embeddings[i]."""
    points = [
        models.PointStruct(
            id=point_id(payload["chunk_id"]),
            vector={
                DENSE: emb.dense,
                SPARSE: models.SparseVector(indices=emb.sparse.indices, values=emb.sparse.values),
            },
            payload=payload,
        )
        for payload, emb in zip(payloads, embeddings, strict=True)
    ]
    for start in range(0, len(points), UPSERT_BATCH):
        client.upsert(collection, points=points[start : start + UPSERT_BATCH], wait=True)


def count_points(client: QdrantClient, collection: str, doc_id: str | None = None) -> int:
    flt = None
    if doc_id:
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
    return client.count(collection, count_filter=flt, exact=True).count
