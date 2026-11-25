"""SQLite database (SQLModel): the document registry.

Tables (Phase 1; users, groups and the audit log come in Phase 3):
  documents          one row per manifest document: title, department,
                     language, allowed_groups... The API will use it to list
                     the documents a user may see.
  indexed_documents  one row per (document, Qdrant collection): the file hash
                     and chunking settings used the last time it was indexed.
                     This makes ingestion idempotent: if the file and the
                     settings did not change, the document is skipped.

Why a database and not just the manifest: the manifest says what SHOULD be
indexed; the registry records what IS indexed, and with which settings.
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, create_engine, select


class Document(SQLModel, table=True):
    __tablename__ = "documents"
    doc_id: str = Field(primary_key=True)
    title: str
    department: str
    language: str
    allowed_groups: list[str] = Field(sa_column=Column(JSON, nullable=False))
    path: str
    source_url: str
    page_count: int | None = None


class IndexedDocument(SQLModel, table=True):
    __tablename__ = "indexed_documents"
    __table_args__ = (UniqueConstraint("doc_id", "collection"),)
    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(index=True)
    collection: str = Field(index=True)
    sha256: str  # fingerprint of the file content
    settings_key: str  # model + chunking settings, e.g. "BAAI/bge-m3|size=400|overlap=0.15|header=1"
    chunk_count: int
    indexed_at: datetime


def get_engine(db_path: Path):
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)  # creates missing tables, keeps existing data
    return engine


def find_indexed(session: Session, doc_id: str, collection: str) -> IndexedDocument | None:
    statement = select(IndexedDocument).where(
        IndexedDocument.doc_id == doc_id, IndexedDocument.collection == collection
    )
    return session.exec(statement).first()


def record_indexed(session: Session, doc_id: str, collection: str, sha256: str,
                   settings_key: str, chunk_count: int) -> None:
    row = find_indexed(session, doc_id, collection) or IndexedDocument(
        doc_id=doc_id, collection=collection, sha256=sha256, settings_key=settings_key,
        chunk_count=chunk_count, indexed_at=datetime.now(timezone.utc),
    )
    row.sha256, row.settings_key, row.chunk_count = sha256, settings_key, chunk_count
    row.indexed_at = datetime.now(timezone.utc)
    session.add(row)
    session.commit()


def save_document(session: Session, document: Document) -> None:
    """Insert or update the registry row of a document."""
    session.merge(document)
    session.commit()
