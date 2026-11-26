"""SQLite database (SQLModel): document registry, users and audit log.

Tables:
  documents          one row per indexed manifest document: title, department,
                     language, allowed_groups... The API lists from it the
                     documents a user may see, and finds the file of a page image.
  indexed_documents  one row per (document, Qdrant collection): the file hash
                     and chunking settings used the last time it was indexed.
                     This makes ingestion idempotent: if the file and the
                     settings did not change, the document is skipped.
  users              (Phase 3) login name, role, departments (groups) and the
                     password hash. Created by `python -m app.seed`.
  audit_log          (Phase 3) one row per question asked through the API: who,
                     when, the question, every chunk retrieved, the passages
                     given to the LLM and which ones the answer cites.

Why a database and not just the manifest: the manifest says what SHOULD be
indexed; the registry records what IS indexed, and with which settings.

The audit log stores the username, role and groups AS THEY WERE when the
question was asked (not a link to the users table): an audit trail must not
change when a user is later edited or deleted.
"""

from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Session, SQLModel, col, create_engine, select


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


class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(unique=True, index=True)
    full_name: str
    role: str  # "employee", "manager" or "admin" (app.auth)
    groups: list[str] = Field(sa_column=Column(JSON, nullable=False))  # departments; "public" is always added
    password_hash: str  # "scrypt$..." (app.auth.hash_password), never the password itself
    created_at: datetime


class AuditEntry(SQLModel, table=True):
    __tablename__ = "audit_log"
    id: int | None = Field(default=None, primary_key=True)
    created_at: datetime = Field(index=True)  # UTC
    username: str = Field(index=True)
    role: str
    # The user's own departments: what a manager's view of the log is based on.
    user_groups: list[str] = Field(sa_column=Column(JSON, nullable=False))
    # The groups given to the access filter: user_groups + public (admin: every group).
    groups: list[str] = Field(sa_column=Column(JSON, nullable=False))
    question: str
    language: str  # detected language of the question
    mode: str  # dense, sparse or hybrid
    reranked: bool
    # Every search candidate, in final order: chunk_id, doc_id, ranks, score, rerank_score.
    retrieved: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    # The passages given to the LLM, [1]..[n]: number, chunk_id, doc_id, cited (true/false).
    passages: list[dict] = Field(sa_column=Column(JSON, nullable=False))
    answer: str | None = None
    answered: bool = False  # False for "I don't know" and for errors
    status: str  # "ok" or "error" (a question runs to the end even if the client leaves)
    error: str | None = None
    seconds: float  # from the start of the search to the end of the answer


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


def forget_document(session: Session, doc_id: str, collection: str) -> None:
    """Remove a document from the registry of one collection, and from
    `documents` once no collection holds it any more (it is no longer indexed)."""
    row = find_indexed(session, doc_id, collection)
    if row:
        session.delete(row)
        session.commit()
    still_indexed = session.exec(select(IndexedDocument).where(IndexedDocument.doc_id == doc_id)).first()
    document = session.get(Document, doc_id)
    if document and not still_indexed:
        session.delete(document)
        session.commit()


def get_user(session: Session, username: str) -> User | None:
    return session.exec(select(User).where(User.username == username)).first()


def save_user(session: Session, username: str, full_name: str, role: str, groups: list[str],
              password_hash: str) -> User:
    """Create the user, or update it (name, role, groups, password) if the username exists."""
    user = get_user(session, username) or User(
        username=username, full_name=full_name, role=role, groups=groups,
        password_hash=password_hash, created_at=datetime.now(timezone.utc),
    )
    user.full_name, user.role, user.groups, user.password_hash = full_name, role, groups, password_hash
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def add_audit_entry(session: Session, entry: AuditEntry) -> int:
    """Store one audit entry and return its id."""
    session.add(entry)
    session.commit()
    return entry.id


def list_audit(session: Session, username: str | None = None) -> list[AuditEntry]:
    """Audit entries, newest first (only this user's if `username` is given)."""
    statement = select(AuditEntry).order_by(col(AuditEntry.id).desc())
    if username:
        statement = statement.where(AuditEntry.username == username)
    return list(session.exec(statement))
