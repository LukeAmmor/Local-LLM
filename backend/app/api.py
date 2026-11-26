"""The web API (FastAPI): login, questions with streamed answers, sources, audit log.

Run:   uv run python -m app.api     (serves on settings api_host:api_port, 127.0.0.1:8100)
Docs:  http://127.0.0.1:8100/docs   (try it: "Authorize" with a token from /api/auth/login)

Endpoints (all under /api, so the Phase 4 UI's dev server forwards a single prefix):
  POST /api/auth/login                     {username, password} -> token   anyone
  GET  /api/me                             profile and effective groups    logged in
  POST /api/ask                            {question} -> streamed answer   logged in
  GET  /api/documents                      the documents I may read        logged in
  GET  /api/documents/{doc_id}/pages/{n}   a PDF page as a PNG image       logged in + allowed
  GET  /api/health                         Qdrant and Ollama status        anyone
  GET  /api/audit                          audit log, newest first         admin; manager (own departments)
  GET  /api/admin/users                    users (never password hashes)   admin
  GET  /api/admin/documents                documents + chunk counts        admin

Access control, in three layers:
  1. Who you are: every endpoint except login and health needs a valid
     token ("Authorization: Bearer <token>"), otherwise 401.
  2. What you may read: your effective groups (app.auth) go into the Qdrant
     filter of every search (app.retrieval), and into the checks of the
     document list and the page images (retrieval.can_read). A document
     outside your groups gives 404, exactly like a document that does not
     exist: the API must not reveal which documents exist (the same idea as
     the "I don't know" sentence, which says "the documents you have access to").
  3. What you may administer: role checks, 403 otherwise.

Streaming (/api/ask) uses Server-Sent Events (SSE): the response stays open
and the server writes events as they happen. Each event is "event: <name>"
and "data: <JSON>" followed by an empty line:
  token  {"text": ...}   a piece of the answer, while the LLM writes it.
  done   {answer, answered, sources, ...}   once, at the end. `answer` is the
         checked answer (invalid citations removed): it replaces the streamed
         text. `sources` are the cited passages only, with their number [n].
  error  {message, audit_id}   something failed (the details are in the audit log).
The data is JSON, so a newline inside the answer cannot break the format.
A browser's EventSource cannot send a POST or an Authorization header, so
the UI (Phase 4) reads this stream with fetch(); `curl -N` shows it live.

One question at a time: a lock covers the whole question (search, rerank,
answer). The 8 GB GPU budget (docs/vram-budget.md) was measured for one
question; a second request waits for the first one. Each question runs in
its own thread, which sends the events to the response through a queue: it
always runs to the end and releases the lock, even when the client leaves
early (see answer_events). Then it writes the question to the audit log
(status "ok" or "error").

Startup (lifespan): load RagPipeline once (embedder on CPU, reranker on GPU,
warm-up), then load the LLM into Ollama (llm_keep_alive = -1 keeps it
loaded). The server accepts requests only after that. One uvicorn worker:
each worker would load its own copy of the models, and the GPU cannot hold
two. The tests give create_app() a fake pipeline instead.
"""

import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import PurePosixPath

import httpx
import jwt
import pymupdf
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sqlmodel import Session, select
from transformers.utils import logging as transformers_logging

from app.auth import (DUMMY_HASH, MIN_SECRET_LENGTH, can_read_audit, create_token, effective_groups, read_token,
                      verify_password)
from app.config import Settings, get_settings
from app.db import AuditEntry, Document, IndexedDocument, User, add_audit_entry, get_engine, get_user, list_audit
from app.generation import load_model
from app.langdetect import detect_language
from app.pipeline import RagPipeline, Result
from app.retrieval import Candidate, can_read

log = logging.getLogger("uvicorn.error")  # uvicorn's logger: our messages appear with its own lines
router = APIRouter(prefix="/api")
bearer = HTTPBearer(auto_error=False)  # reads "Authorization: Bearer <token>"; we answer the 401 ourselves


class LoginRequest(BaseModel):
    username: str
    password: str


class AskRequest(BaseModel):
    question: str


# --- Dependencies: what FastAPI computes before calling an endpoint ---

def get_session(request: Request) -> Iterator[Session]:
    """A database session for one request, closed after it."""
    with Session(request.app.state.engine) as session:
        yield session


def unauthorized(detail: str) -> HTTPException:
    return HTTPException(401, detail, headers={"WWW-Authenticate": "Bearer"})


def current_user(request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
                 session: Session = Depends(get_session)) -> User:
    """The logged-in user, read from the database: 401 without a valid token."""
    if credentials is None:
        raise unauthorized("not logged in")
    try:
        username = read_token(credentials.credentials, request.app.state.settings.jwt_secret.get_secret_value())
    except jwt.InvalidTokenError:
        raise unauthorized("invalid or expired token") from None
    user = get_user(session, username)
    if user is None:  # deleted after the token was made
        raise unauthorized("unknown user")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "admin only")
    return user


def audit_reader(user: User = Depends(current_user)) -> User:
    if user.role not in ("admin", "manager"):
        raise HTTPException(403, "admin or manager only")
    return user


def groups_of(request: Request, user: User) -> list[str]:
    """The user's effective groups: what the access filter and the document checks use."""
    return effective_groups(user, request.app.state.pipeline.known_groups, request.app.state.settings.public_group)


# --- What the endpoints return ---

def user_info(user: User, groups: list[str]) -> dict:
    return {"username": user.username, "full_name": user.full_name, "role": user.role,
            "groups": user.groups, "effective_groups": groups}


def document_info(document: Document) -> dict:
    return {"doc_id": document.doc_id, "title": document.title, "department": document.department,
            "language": document.language, "format": PurePosixPath(document.path).suffix.lstrip(".").lower(),
            "page_count": document.page_count, "source_url": document.source_url}


def source_info(number: int, c: Candidate) -> dict:
    """A cited passage, as the UI shows it. PDF: page_start/page_end; DOCX: no pages, the section."""
    p = c.payload
    return {"number": number, "doc_id": p["doc_id"], "title": p["title"], "section": p["section"],
            "page_start": p["page_start"], "page_end": p["page_end"], "language": p["language"],
            "text": p["text"], "rerank_score": c.rerank_score}


def sse(event: str, data: dict) -> str:
    """One Server-Sent Event: its name and its data as one line of JSON, then an empty line."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# --- Endpoints ---

@router.post("/auth/login")
def login(body: LoginRequest, request: Request, session: Session = Depends(get_session)) -> dict:
    """Check the password and return a login token. Same answer for an unknown user and a wrong password."""
    user = get_user(session, body.username)
    password_ok = verify_password(body.password, user.password_hash if user else DUMMY_HASH)
    if user is None or not password_ok:
        raise HTTPException(401, "wrong username or password")
    s = request.app.state.settings
    token = create_token(user.username, s.jwt_secret.get_secret_value(), s.jwt_expire_minutes)
    return {"access_token": token, "token_type": "bearer", "user": user_info(user, groups_of(request, user))}


@router.get("/me")
def me(request: Request, user: User = Depends(current_user)) -> dict:
    return user_info(user, groups_of(request, user))


@router.post("/ask")
def ask(body: AskRequest, request: Request, user: User = Depends(current_user)) -> StreamingResponse:
    """Answer a question from the documents the user may read; the answer is streamed (SSE)."""
    s = request.app.state.settings
    question = body.question.strip()
    if not question or len(question) > s.question_max_chars:
        raise HTTPException(422, f"the question must have 1 to {s.question_max_chars} characters")
    # Plain copies: the stream runs after this request's database session is closed.
    events = answer_events(request.app.state, user.username, user.role, list(user.groups),
                           groups_of(request, user), question)
    return StreamingResponse(events, media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


def answer_events(state, username: str, role: str, own_groups: list[str], groups: list[str],
                  question: str) -> Iterator[str]:
    """The SSE events of one question, read from a queue filled by run_question() in its own thread.

    Why a thread and a queue, and not the lock and the pipeline inside this
    generator: when the client leaves, Starlette stops reading the generator
    and leaves it paused at a `yield`. A generator that held the lock would
    then keep it, and every later question would wait for ever (measured).
    The thread does not depend on the client: it always finishes the
    question, writes the audit row and releases the lock.
    """
    events: queue.Queue[str | None] = queue.Queue()
    threading.Thread(target=run_question, daemon=True,
                     args=(state, events, username, role, own_groups, groups, question)).start()
    while (event := events.get()) is not None:  # None = end of the stream
        yield event


def run_question(state, events: queue.Queue, username: str, role: str, own_groups: list[str],
                 groups: list[str], question: str) -> None:
    """Answer one question, putting its SSE events in `events`, then None. Runs in its own thread."""
    try:
        with state.question_lock:  # one question at a time (GPU memory budget)
            start = time.perf_counter()
            result, error = None, None
            try:
                result = state.pipeline.answer(question, groups, generate=False)  # embed, search, rerank
                for piece in state.pipeline.generate_stream(result):  # the LLM writes the answer
                    events.put(sse("token", {"text": piece}))
            except Exception as exc:  # noqa: BLE001 (Qdrant or Ollama down, prompt too long...)
                error = f"{type(exc).__name__}: {exc}"
                log.exception("question failed")
            entry = audit_entry(state.settings, username, role, own_groups, groups, question,
                                result, error, time.perf_counter() - start)
            with Session(state.engine) as session:
                audit_id = add_audit_entry(session, entry)
        if error is None:
            events.put(sse("done", {
                "answer": result.answer, "answered": result.answered, "language": result.language,
                "sources": [source_info(n, c) for n, c in result.sources],
                "invalid_citations": result.invalid_citations, "warnings": result.warnings,
                "timings": {step: round(seconds, 3) for step, seconds in result.timings.items()},
                "audit_id": audit_id,
            }))
        else:
            events.put(sse("error", {"message": "the answer could not be generated (the error is in the audit log)",
                                     "audit_id": audit_id}))
    finally:
        events.put(None)  # always end the stream, even if writing the audit row failed


def audit_entry(s: Settings, username: str, role: str, own_groups: list[str], groups: list[str], question: str,
                result: Result | None, error: str | None, seconds: float) -> AuditEntry:
    """The audit log row of one question (`error` is None when it was answered)."""
    status = "ok" if error is None else "error"
    entry = AuditEntry(
        created_at=datetime.now(timezone.utc), username=username, role=role, user_groups=own_groups,
        groups=groups, question=question, language=detect_language(question), mode=s.retrieval_mode,
        reranked=s.rerank_enabled, retrieved=[], passages=[], status=status, error=error, seconds=round(seconds, 3),
    )
    if result is not None:  # None when the search itself failed
        entry.language, entry.mode, entry.reranked = result.language, result.mode, result.reranked
        entry.retrieved = [{"chunk_id": c.payload["chunk_id"], "doc_id": c.payload["doc_id"], "ranks": c.ranks,
                            "score": c.score, "rerank_score": c.rerank_score} for c in result.candidates]
        entry.passages = [{"number": n, "chunk_id": p.payload["chunk_id"], "doc_id": p.payload["doc_id"],
                           "cited": p.cited} for n, p in enumerate(result.passages, start=1)]
        entry.answer = result.answer
        entry.answered = result.answered and status == "ok"
    return entry


@router.get("/documents")
def my_documents(request: Request, user: User = Depends(current_user),
                 session: Session = Depends(get_session)) -> list[dict]:
    """The indexed documents the user may read (their groups + public)."""
    groups = groups_of(request, user)
    documents = session.exec(select(Document).order_by(Document.department, Document.doc_id)).all()
    return [document_info(d) for d in documents if can_read(d.allowed_groups, groups)]


@router.get("/documents/{doc_id}/pages/{page}", response_class=Response,
            responses={200: {"content": {"image/png": {}}, "description": "The page as a PNG image."}})
def page_image(doc_id: str, page: int, request: Request, user: User = Depends(current_user),
               session: Session = Depends(get_session)) -> Response:
    """A PDF page (numbered from 1) as a PNG image, to show a source."""
    s = request.app.state.settings
    document = session.get(Document, doc_id)
    if document is None or not can_read(document.allowed_groups, groups_of(request, user)):
        raise HTTPException(404, "document not found")  # the same answer: existence is not revealed
    if PurePosixPath(document.path).suffix.lower() != ".pdf":
        raise HTTPException(400, "this document has no pages (DOCX): its citations show the section")
    path = s.raw_dir / document.path
    if not path.exists():
        raise HTTPException(404, "the file of this document is missing on the server")
    with pymupdf.open(path) as pdf:
        if not 1 <= page <= pdf.page_count:
            raise HTTPException(404, "page not found")
        png = pdf[page - 1].get_pixmap(dpi=s.page_image_dpi).tobytes("png")
    # no-store: the browser does not keep confidential pages in its disk cache.
    return Response(png, media_type="image/png", headers={"Cache-Control": "no-store"})


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Are Qdrant and Ollama ready? No login needed; 503 if something is missing.

    qdrant: reachable and the collection exists. ollama: reachable and the
    LLM is pulled. The models are loaded before the server accepts requests.
    """
    s = request.app.state.settings
    checks = {"qdrant": False, "ollama": False}
    try:
        checks["qdrant"] = request.app.state.pipeline.qdrant.collection_exists(s.qdrant_collection)
    except Exception:  # noqa: BLE001 (any connection error means "not ready")
        pass
    try:
        models = httpx.get(f"{s.ollama_url}/api/tags", timeout=5).json()["models"]
        checks["ollama"] = any(m["name"] == s.llm_model for m in models)
    except httpx.HTTPError:
        pass
    ok = all(checks.values())
    return JSONResponse({"status": "ok" if ok else "degraded", **checks}, status_code=200 if ok else 503)


@router.get("/audit")
def audit_log(request: Request, viewer: User = Depends(audit_reader), session: Session = Depends(get_session),
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
              username: str | None = None) -> dict:
    """The audit log, newest first: every entry for an admin, the manager's own departments for a manager.

    Filtered in Python: simple, and fine for a demo-sized log. A very large
    log would need the groups in a table of their own, to filter in SQL.
    """
    public = request.app.state.settings.public_group
    visible = [e for e in list_audit(session, username) if can_read_audit(viewer, e, public)]
    return {"total": len(visible), "entries": visible[offset : offset + limit]}


@router.get("/admin/users")
def admin_users(request: Request, admin: User = Depends(admin_user),
                session: Session = Depends(get_session)) -> list[dict]:
    users = session.exec(select(User).order_by(User.username)).all()
    return [user_info(u, groups_of(request, u)) for u in users]


@router.get("/admin/documents")
def admin_documents(request: Request, admin: User = Depends(admin_user),
                    session: Session = Depends(get_session)) -> list[dict]:
    """Every indexed document, with its groups and its chunk count in the current collection."""
    s = request.app.state.settings
    indexed = {r.doc_id: r for r in session.exec(
        select(IndexedDocument).where(IndexedDocument.collection == s.qdrant_collection))}
    rows = []
    for d in session.exec(select(Document).order_by(Document.department, Document.doc_id)):
        record = indexed.get(d.doc_id)
        rows.append({**document_info(d), "allowed_groups": d.allowed_groups,
                     "chunks": record.chunk_count if record else 0,
                     "indexed_at": record.indexed_at if record else None})
    return rows


# --- The app ---

def preload_llm(s: Settings) -> None:
    try:
        load_model(s.ollama_url, s.llm_model, s.llm_num_ctx, s.llm_keep_alive, s.llm_timeout_s)
        log.info("%s loaded in Ollama (keep_alive %s).", s.llm_model, s.llm_keep_alive)
    except httpx.HTTPError as exc:
        log.warning("Could not load %s in Ollama (%s): the first question will load it.", s.llm_model, exc)


def create_app(settings: Settings, pipeline: RagPipeline | None = None) -> FastAPI:
    """The FastAPI app. Without `pipeline`, the real one is loaded at startup (the tests give a fake one)."""
    if len(settings.jwt_secret.get_secret_value()) < MIN_SECRET_LENGTH:
        raise ValueError(f"set RAG_JWT_SECRET in .env (at least {MIN_SECRET_LENGTH} characters). Generate one with:\n"
                         "  uv run python -c \"import secrets; print(secrets.token_urlsafe(32))\"")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            log.info("Loading the models: embedder on %s, reranker on %s...",
                     settings.embed_device, settings.rerank_device)
            app.state.pipeline = RagPipeline(settings)
            preload_llm(settings)
        yield

    app = FastAPI(title="Local RAG assistant", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = get_engine(settings.db_path)
    app.state.pipeline = pipeline
    app.state.question_lock = threading.Lock()
    app.include_router(router)
    return app


def main() -> int:
    s = get_settings()
    transformers_logging.disable_progress_bar()  # no "Loading weights" bars in the server log
    try:
        app = create_app(s)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    uvicorn.run(app, host=s.api_host, port=s.api_port)  # one worker: see the module docstring
    return 0


if __name__ == "__main__":
    sys.exit(main())
