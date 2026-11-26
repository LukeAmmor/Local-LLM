"""The question-answering pipeline: embed -> search -> rerank -> generate -> check citations.

What: `RagPipeline` loads the models once (embedder, reranker), then
`answer(question, groups)` returns a `Result` with everything the later
phases need: the answer, ALL the passages given to the LLM (each marked
cited or not; the RAGAS context metrics of Phase 5 need all of them), the
search candidates with their ranks and scores, warnings and timings.
Used by the CLI (app.ask), later by the API (Phase 3) and the evaluation
(Phase 5).

Steps of `answer()`:
  1. Embed the question with bge-m3 (on the CPU: setting embed_device).
  2. Search Qdrant (app.retrieval) for the user's groups + the public group:
     the best retrieval_top_k (30) chunks the user may read.
  3. Rerank them (app.rerank, on the GPU) and sort by rerank score.
  4. Keep the best context_passages (5): the LLM's context, numbered [1]..[5].
  5. Generate the answer (app.generation) and check its citations: invalid
     numbers are removed and reported, cited passages are marked.
Each stage can be switched for the experiments: search mode dense / sparse
/ hybrid, reranking on or off, generation on or off (retrieval only).

When no passage is found (e.g. no document for the user's groups), the
answer is "I don't know" and the LLM is not called.
"""

import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from app.chunking import with_header
from app.config import Settings
from app.embeddings import Embedder
from app.generation import LLMReply, build_messages, chat, check_citations, idk_sentence, is_refusal
from app.langdetect import detect_language
from app.manifest import load_manifest
from app.rerank import Reranker
from app.retrieval import Candidate, check_access, search, user_groups


@dataclass
class Result:
    question: str
    language: str  # detected language of the question: "en", "fr" or "unknown"
    groups: list[str]  # groups used by the access filter (the public group included)
    mode: str  # "dense", "sparse" or "hybrid"
    reranked: bool
    candidates: list[Candidate]  # every search result, in final order (after reranking if on)
    passages: list[Candidate]  # the first context_passages candidates: the LLM's context, [1]..[n]
    answer: str | None = None  # None when generation is off
    answered: bool = False  # False for "I don't know" (and when generation is off)
    invalid_citations: list[int] = field(default_factory=list)  # removed from the answer
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    llm: LLMReply | None = None

    @property
    def sources(self) -> list[tuple[int, Candidate]]:
        """The cited passages with their numbers: what the CLI (and later the UI) shows."""
        return [(n, p) for n, p in enumerate(self.passages, start=1) if p.cited]


class RagPipeline:
    def __init__(self, settings: Settings, load_reranker: bool = True):
        s = self.s = settings
        self.qdrant = QdrantClient(url=s.qdrant_url, timeout=30)
        self.embedder = Embedder(s.embed_model, s.embed_device, s.embed_fp16, s.embed_batch_size, s.embed_max_length)
        self.reranker = None
        if load_reranker:
            self.reranker = Reranker(s.rerank_model, s.rerank_device, s.rerank_fp16,
                                     s.rerank_batch_size, s.rerank_max_length)
        # Warm-up: the first run of a model is several times slower (memory
        # allocation, GPU setup). Doing it here, once, keeps the per-question
        # timings comparable between runs.
        self.embedder.encode(["warm-up"])
        if self.reranker:
            self.reranker.score("warm-up", ["warm-up"])
        # Groups that some indexed document uses: to warn about typos in a user's groups.
        manifest = load_manifest(s.manifest_path)
        self.known_groups = {g for d in manifest.documents if not d.exclude for g in d.allowed_groups}

    def rerank_text(self, payload: dict) -> str:
        """The passage as the reranker reads it: with the "Title > Section" header if chunk_add_header."""
        if self.s.chunk_add_header:
            return with_header(payload["title"], payload["section"], payload["text"])
        return payload["text"]

    def answer(self, question: str, groups: list[str], mode: str | None = None, rerank: bool | None = None,
               generate: bool = True, collection: str | None = None) -> Result:
        s = self.s
        mode = mode or s.retrieval_mode
        rerank = s.rerank_enabled if rerank is None else rerank
        collection = collection or s.qdrant_collection
        groups_used = user_groups(groups, s.public_group)
        warnings = [f"group '{g}' is not used by any document" for g in groups if g not in self.known_groups]
        timings = {}

        start = time.perf_counter()
        embedding = self.embedder.encode([question])[0]
        timings["embed"] = time.perf_counter() - start

        start = time.perf_counter()
        candidates = search(self.qdrant, collection, embedding, groups_used, mode, s.retrieval_top_k, s.rrf_k)
        check_access(candidates, groups_used)
        timings["search"] = time.perf_counter() - start

        if rerank:
            if self.reranker is None:
                raise ValueError("reranking was requested but the pipeline was created without the reranker")
            start = time.perf_counter()
            scores = self.reranker.score(question, [self.rerank_text(c.payload) for c in candidates])
            for candidate, score in zip(candidates, scores, strict=True):
                candidate.rerank_score = score
            candidates.sort(key=lambda c: c.rerank_score, reverse=True)
            timings["rerank"] = time.perf_counter() - start

        result = Result(question=question, language=detect_language(question), groups=groups_used,
                        mode=mode, reranked=rerank, candidates=candidates,
                        passages=candidates[: s.context_passages], warnings=warnings, timings=timings)
        if generate:
            self.generate(result)
        return result

    def generate(self, result: Result) -> None:
        """Step 5: ask the LLM, then check the citations of its answer (fills `result`)."""
        s = self.s
        if not result.passages:
            result.answer = idk_sentence(result.language)
            result.warnings.append("no passage found for these groups: the LLM was not called")
            return

        start = time.perf_counter()
        messages = build_messages(result.question, [p.payload for p in result.passages], result.language)
        reply = chat(s.ollama_url, s.llm_model, messages, s.llm_num_ctx, s.llm_num_predict,
                     s.llm_temperature, s.llm_keep_alive, s.llm_timeout_s)
        result.timings["generation"] = time.perf_counter() - start
        result.llm = reply

        text, cited, invalid = check_citations(reply.text, len(result.passages))
        for number, passage in enumerate(result.passages, start=1):
            passage.cited = number in cited
        result.answer = text
        result.answered = not is_refusal(text, cited)
        result.invalid_citations = invalid
        if invalid:
            result.warnings.append(f"invalid citations removed: {', '.join(f'[{n}]' for n in invalid)} "
                                   f"(there are {len(result.passages)} passages)")
        if result.answered and not cited:
            result.warnings.append("the answer cites no passage")
        if reply.done_reason == "length":
            result.warnings.append(f"the answer was cut at llm_num_predict = {s.llm_num_predict} tokens")
