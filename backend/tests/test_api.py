"""Tests for app.api: login, access control through the API, audit log.

No models, no Ollama, no Docker: everything runs in `uv run pytest`.
The app gets a FakePipeline: the real RagPipeline code (search with the
access filter, citation check), but with an in-memory Qdrant (the local mode
of qdrant-client), a constant fake embedding and a fake LLM. All the chunks
have the SAME vectors, so only the access filter decides what a user gets.
The documents are tiny PDFs made with PyMuPDF in a temporary folder,
registered in a temporary SQLite database.
"""

import json
from datetime import datetime, timedelta, timezone

import jwt
import pymupdf
import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from sqlmodel import Session

from app.api import answer_events, create_app
from app.auth import create_token, hash_password
from app.config import Settings
from app.db import Document, get_engine, save_user
from app.embeddings import Embedding, SparseVector
from app.generation import LLMReply
from app.pipeline import RagPipeline
from app.vectorstore import ensure_collection, upsert_chunks

# The in-memory Qdrant warns that payload indexes do nothing there (they only make filters faster).
pytestmark = pytest.mark.filterwarnings("ignore:Payload indexes have no effect:UserWarning")

SECRET = "test-secret-" + "x" * 40
PASSWORD = "demo-password"
PASSWORD_HASH = hash_password(PASSWORD)  # once: scrypt is slow on purpose
EMBEDDING = Embedding(dense=[1.0] * 1024, sparse=SparseVector(indices=[1, 2], values=[1.0, 0.5]))

DOCUMENTS = [  # doc_id, group, path in raw/, pages (0: a DOCX, registered but no file needed)
    ("hr-policy", "hr", "hr/hr-policy.pdf", 2),
    ("procurement-rules", "procurement", "procurement/procurement-rules.pdf", 1),
    ("procurement-regs-fr", "procurement", "procurement/procurement-regs-fr.docx", 0),
    ("public-guide", "public", "public/public-guide.pdf", 1),
]
USERS = [  # username, role, groups
    ("hr.officer", "employee", ["hr"]),
    ("hr.manager", "manager", ["hr"]),
    ("procurement.officer", "employee", ["procurement"]),
    ("admin", "admin", []),
]
HR_READS = {"hr-policy", "public-guide"}
PROCUREMENT_READS = {"procurement-rules", "procurement-regs-fr", "public-guide"}


def chunk_payload(doc_id: str, group: str, path: str, index: int) -> dict:
    page = 1 if path.endswith(".pdf") else None
    return {"chunk_id": f"{doc_id}:{index}", "doc_id": doc_id, "title": f"Title of {doc_id}", "section": "Rules",
            "page_start": page, "page_end": page, "language": "en", "allowed_groups": [group],
            "text": f"Text of {doc_id}, chunk {index}."}


class FakeEmbedder:
    def encode(self, texts):
        return [EMBEDDING for _ in texts]


class FakePipeline(RagPipeline):
    """RagPipeline without models: the same search, access filter and citation code."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.qdrant = QdrantClient(":memory:")
        self.embedder = FakeEmbedder()
        self.reranker = None
        self.known_groups = {group for _, group, _, _ in DOCUMENTS}
        ensure_collection(self.qdrant, settings.qdrant_collection)
        payloads = [chunk_payload(doc_id, group, path, i) for doc_id, group, path, _ in DOCUMENTS for i in range(2)]
        upsert_chunks(self.qdrant, settings.qdrant_collection, payloads, [EMBEDDING] * len(payloads))


@pytest.fixture
def prompts(monkeypatch):
    """Replace the LLM: it answers "Answer [1]." and keeps the prompts it receives."""
    received = []

    def fake_chat_stream(ollama_url, model, messages, *args):
        received.append(messages[1]["content"])  # the user message: numbered passages + question
        yield "token", "Answer "
        yield "token", "[1]."
        yield "done", LLMReply(text="Answer [1].", prompt_tokens=0, answer_tokens=0, prompt_seconds=0.0,
                               answer_seconds=0.0, load_seconds=0.0, done_reason="stop")

    monkeypatch.setattr("app.pipeline.chat_stream", fake_chat_stream)
    return received


@pytest.fixture
def client(tmp_path, prompts):
    # _env_file=None: the real .env is not read. ollama_url points to a closed
    # port: nothing in these tests may reach a real LLM.
    settings = Settings(_env_file=None, data_dir=tmp_path, jwt_secret=SECRET, rerank_enabled=False,
                        qdrant_collection="test", ollama_url="http://127.0.0.1:9")
    engine = get_engine(settings.db_path)
    with Session(engine) as session:
        for doc_id, group, path, pages in DOCUMENTS:
            if pages:
                file = settings.raw_dir / path
                file.parent.mkdir(parents=True, exist_ok=True)
                with pymupdf.open() as pdf:
                    for n in range(1, pages + 1):
                        pdf.new_page().insert_text((72, 72), f"{doc_id}, page {n}")
                    pdf.save(str(file))
            session.add(Document(doc_id=doc_id, title=f"Title of {doc_id}", department=group, language="en",
                                 allowed_groups=[group], path=path, source_url="", page_count=pages or None))
        session.commit()
        for username, role, groups in USERS:
            save_user(session, username, username, role, groups, PASSWORD_HASH)
    return TestClient(create_app(settings, FakePipeline(settings)))


def login(client: TestClient, username: str) -> dict:
    """Log in and return the Authorization header."""
    response = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    assert response.status_code == 200
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def ask(client: TestClient, username: str, question: str = "What is the procedure?") -> list[tuple[str, dict]]:
    """Ask a question as this user; return the SSE events as (name, data)."""
    response = client.post("/api/ask", json={"question": question}, headers=login(client, username))
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = []
    for block in response.text.strip().split("\n\n"):
        name, data = block.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def audit(client: TestClient, username: str = "admin", query: str = "") -> list[dict]:
    response = client.get(f"/api/audit{query}", headers=login(client, username))
    assert response.status_code == 200
    return response.json()["entries"]


# --- Login and authentication ---

def test_login_returns_a_token_and_the_profile(client):
    response = client.post("/api/auth/login", json={"username": "hr.officer", "password": PASSWORD})
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["user"] == {"username": "hr.officer", "full_name": "hr.officer", "role": "employee",
                            "groups": ["hr"], "effective_groups": ["hr", "public"]}
    me = client.get("/api/me", headers={"Authorization": f"Bearer {body['access_token']}"})
    assert me.json()["effective_groups"] == ["hr", "public"]


def test_wrong_password_and_unknown_user_get_the_same_answer(client):
    wrong = client.post("/api/auth/login", json={"username": "hr.officer", "password": "wrong"})
    unknown = client.post("/api/auth/login", json={"username": "nobody", "password": PASSWORD})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # does not reveal which usernames exist


PROTECTED = [
    ("GET", "/api/me"), ("POST", "/api/ask"), ("GET", "/api/documents"),
    ("GET", "/api/documents/hr-policy/pages/1"), ("GET", "/api/audit"),
    ("GET", "/api/admin/users"), ("GET", "/api/admin/documents"),
]


def bad_tokens() -> dict[str, str | None]:
    now = datetime.now(timezone.utc)
    return {
        "no token": None,
        "garbage": "not-a-token",
        "expired": create_token("hr.officer", SECRET, 60, now=now - timedelta(hours=2)),
        "other secret": create_token("admin", "another-secret-" + "y" * 40, 60),
        "unsigned": jwt.encode({"sub": "admin", "exp": now + timedelta(hours=1)}, None, algorithm="none"),
        "deleted user": create_token("ghost", SECRET, 60),
    }


@pytest.mark.parametrize("method, path", PROTECTED)
def test_unauthenticated_requests_are_refused(client, prompts, method, path):
    for name, token in bad_tokens().items():
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        body = {"question": "What is the procedure?"} if method == "POST" else None
        response = client.request(method, path, headers=headers, json=body)
        assert response.status_code == 401, name
    assert prompts == []  # the LLM was never called


def test_health_needs_no_login(client):
    response = client.get("/api/health")
    assert response.status_code == 503  # no Ollama in the tests
    assert response.json() == {"status": "degraded", "qdrant": True, "ollama": False}


def test_app_refuses_to_start_without_a_long_secret(tmp_path):
    with pytest.raises(ValueError, match="RAG_JWT_SECRET"):
        create_app(Settings(_env_file=None, data_dir=tmp_path, jwt_secret="short"))


# --- Access control: chunks, documents, page images ---

def test_hr_user_never_gets_procurement_chunks(client, prompts):
    events = ask(client, "hr.officer")
    assert [name for name, _ in events] == ["token", "token", "done"]
    done = events[-1][1]
    assert done["answer"] == "Answer [1]."
    assert len(done["sources"]) == 1 and done["sources"][0]["doc_id"] in HR_READS
    # The LLM saw hr and public passages only.
    assert "Text of hr-policy" in prompts[0] and "procurement" not in prompts[0]
    # The audit log: everything the hr user may read was retrieved, nothing else.
    entry = audit(client)[0]
    assert (entry["username"], entry["question"], entry["status"]) == ("hr.officer", "What is the procedure?", "ok")
    assert entry["groups"] == ["hr", "public"]
    assert {c["doc_id"] for c in entry["retrieved"]} == HR_READS
    assert {p["doc_id"] for p in entry["passages"]} <= HR_READS
    assert [p["cited"] for p in entry["passages"]].count(True) == 1
    assert entry["id"] == done["audit_id"]


def test_procurement_user_never_gets_hr_chunks(client, prompts):
    done = ask(client, "procurement.officer")[-1][1]
    assert all(s["doc_id"] in PROCUREMENT_READS for s in done["sources"])
    assert "Text of procurement-rules" in prompts[0] and "hr-policy" not in prompts[0]
    assert {c["doc_id"] for c in audit(client)[0]["retrieved"]} == PROCUREMENT_READS


def test_document_list_follows_the_groups(client):
    def doc_ids(username):
        return {d["doc_id"] for d in client.get("/api/documents", headers=login(client, username)).json()}

    assert doc_ids("hr.officer") == HR_READS
    assert doc_ids("procurement.officer") == PROCUREMENT_READS
    assert doc_ids("admin") == {doc_id for doc_id, _, _, _ in DOCUMENTS}


def test_hr_user_never_gets_procurement_page_images(client):
    hr = login(client, "hr.officer")
    forbidden = client.get("/api/documents/procurement-rules/pages/1", headers=hr)
    missing = client.get("/api/documents/no-such-document/pages/1", headers=hr)
    assert forbidden.status_code == missing.status_code == 404
    assert forbidden.json() == missing.json()  # a forbidden document looks exactly like a missing one
    assert client.get("/api/documents/procurement-regs-fr/pages/1", headers=hr).status_code == 404
    admin = client.get("/api/documents/procurement-rules/pages/1", headers=login(client, "admin"))
    assert admin.status_code == 200


def test_page_images(client):
    hr = login(client, "hr.officer")
    page = client.get("/api/documents/hr-policy/pages/2", headers=hr)
    assert page.status_code == 200
    assert page.headers["content-type"] == "image/png" and page.content.startswith(b"\x89PNG")
    assert page.headers["cache-control"] == "no-store"
    assert client.get("/api/documents/hr-policy/pages/0", headers=hr).status_code == 404
    assert client.get("/api/documents/hr-policy/pages/3", headers=hr).status_code == 404
    docx = client.get("/api/documents/procurement-regs-fr/pages/1", headers=login(client, "procurement.officer"))
    assert docx.status_code == 400  # DOCX: no pages, cited by section


# --- Roles: admin endpoints and audit log ---

@pytest.mark.parametrize("path", ["/api/admin/users", "/api/admin/documents"])
def test_admin_endpoints_refuse_non_admins(client, path):
    for username in ("hr.officer", "hr.manager", "procurement.officer"):
        assert client.get(path, headers=login(client, username)).status_code == 403, username
    assert client.get(path, headers=login(client, "admin")).status_code == 200


def test_user_list_never_shows_password_hashes(client):
    users = client.get("/api/admin/users", headers=login(client, "admin")).json()
    assert {u["username"] for u in users} == {username for username, _, _ in USERS}
    assert "password_hash" not in json.dumps(users) and "scrypt" not in json.dumps(users)


def test_employee_cannot_read_the_audit_log(client):
    assert client.get("/api/audit", headers=login(client, "hr.officer")).status_code == 403


def test_manager_reads_the_audit_log_of_their_department_only(client, prompts):
    for username in ("hr.officer", "procurement.officer", "admin"):
        ask(client, username)
    assert {e["username"] for e in audit(client, "hr.manager")} == {"hr.officer"}
    assert audit(client, "hr.manager", "?username=procurement.officer") == []
    assert {e["username"] for e in audit(client, "admin")} == {"hr.officer", "procurement.officer", "admin"}


# --- Errors ---

def test_question_length_is_checked(client, prompts):
    hr = login(client, "hr.officer")
    assert client.post("/api/ask", json={"question": "   "}, headers=hr).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 1001}, headers=hr).status_code == 422
    assert prompts == []


def test_client_leaving_early_does_not_block_the_next_questions(client, prompts):
    # Read one event of the stream, then leave without closing it, like a
    # browser tab closed during the answer.
    state = client.app.state
    stream = answer_events(state, "hr.officer", "employee", ["hr"], ["hr", "public"], "What is the procedure?")
    assert next(stream).startswith("event: token")
    # The question finishes on its own thread and releases the lock.
    assert state.question_lock.acquire(timeout=10)
    state.question_lock.release()
    entry = audit(client)[0]
    assert (entry["username"], entry["status"], entry["answer"]) == ("hr.officer", "ok", "Answer [1].")
    assert ask(client, "hr.officer")[-1][0] == "done"


def test_llm_failure_gives_an_error_event_and_an_audit_entry(client, monkeypatch):
    def broken_llm(*args):
        raise RuntimeError("Ollama is down")
        yield  # makes this function a generator, like chat_stream

    monkeypatch.setattr("app.pipeline.chat_stream", broken_llm)
    name, data = ask(client, "hr.officer")[-1]
    assert name == "error" and "Ollama" not in data["message"]  # details only in the audit log
    entry = audit(client)[0]
    assert entry["status"] == "error" and "Ollama is down" in entry["error"]
    assert entry["id"] == data["audit_id"] and not entry["answered"]
