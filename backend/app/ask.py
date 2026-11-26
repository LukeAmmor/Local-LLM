"""Ask a question from the command line: search -> rerank -> answer with citations.

Usage:
    uv run python -m app.ask "Que faire si un collègue me harcèle ?" --groups hr
    uv run python -m app.ask "..." --groups procurement --mode dense --no-rerank
    uv run python -m app.ask "..." --groups hr --no-generate --verbose   # retrieval only

No login here: this is a local tool for the experiments, and the groups are
given directly. Users, logins and the audit log are in the API (app.api).
The public group is always added.

Options (the defaults come from the settings, backend/app/config.py):
  --mode dense|sparse|hybrid   search mode (setting retrieval_mode)
  --rerank / --no-rerank       rerank the candidates (setting rerank_enabled)
  --no-generate                stop after retrieval: show the passages the LLM would get
  --collection NAME            another Qdrant collection (experiments, e.g. kb_c256)
  --verbose                    every candidate with its ranks and scores, and full passage texts
Other values (top_k, number of passages, devices...) are settings too, and can be
changed for one run with RAG_* environment variables, e.g. RAG_CONTEXT_PASSAGES=3.

Output: the answer, the cited sources (title, page or section, text), the
warnings, and the time of each step. The LLM line shows how fast the prompt
was read: about 2,700 tokens/s when healthy, about 300 when Windows pages
GPU memory to RAM (docs/vram-budget.md). Ollama reuses the start of the
previous prompt when it is the same (the system prompt, or the same question
asked again), so the speed is only meaningful for a new question.
"""

import argparse
import sys
import time

import httpx
from transformers.utils import logging as transformers_logging

from app.config import get_settings
from app.generation import location
from app.pipeline import RagPipeline, Result
from app.retrieval import Candidate

PREVIEW_CHARS = 300  # passage text shown without --verbose


def preview(text: str, full: bool) -> str:
    if full:
        return text
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= PREVIEW_CHARS else one_line[:PREVIEW_CHARS] + "..."


def print_passage(number: int, c: Candidate, full: bool) -> None:
    p = c.payload
    where = location(p)
    print(f"[{number}] {p['title']}" + (f", {where}" if where else ""))
    if p["page_start"] is not None and p["section"]:  # PDF: the page is in the title line, add the section
        print(f"    section: {p['section']}")
    scores = f"search score {c.score:.4f}" + (f", rerank {c.rerank_score:.3f}" if c.rerank_score is not None else "")
    print(f"    {p['doc_id']} | {p['language']} | {scores}")
    for line in preview(p["text"], full).splitlines():
        print(f"    {line}")


def print_candidates(result: Result) -> None:
    """Every candidate, in final order, with its rank in each search (-- = not found by that search)."""
    print(f"\nCandidates ({len(result.candidates)}; * = given to the LLM):")
    print(f"  {'#':>3}  {'rerank':>6}  {'score':>7}  {'dense':>5}  {'sparse':>6}  {'doc_id':<28} where")
    for i, c in enumerate(result.candidates, start=1):
        mark = "*" if i <= len(result.passages) else " "
        rerank = f"{c.rerank_score:.3f}" if c.rerank_score is not None else "--"
        dense, sparse = c.ranks.get("dense", "--"), c.ranks.get("sparse", "--")
        print(f"{mark} {i:>3}  {rerank:>6}  {c.score:>7.4f}  {dense:>5}  {sparse:>6}  "
              f"{c.payload['doc_id']:<28} {location(c.payload)[:40]}")


def print_timings(result: Result, load_seconds: float) -> None:
    steps = " | ".join(f"{name} {seconds:.2f} s" for name, seconds in result.timings.items())
    print(f"\nTimings: {steps} | total {sum(result.timings.values()):.2f} s "
          f"(models loaded once in {load_seconds:.1f} s)")
    if result.llm:
        r = result.llm
        prompt_speed = r.prompt_tokens / r.prompt_seconds if r.prompt_seconds else 0
        answer_speed = r.answer_tokens / r.answer_seconds if r.answer_seconds else 0
        print(f"LLM: prompt {r.prompt_tokens} tokens read in {r.prompt_seconds:.2f} s ({prompt_speed:,.0f} tokens/s) | "
              f"answer {r.answer_tokens} tokens in {r.answer_seconds:.2f} s ({answer_speed:.0f} tokens/s) | "
              f"model load {r.load_seconds:.1f} s")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # answers contain accents; Windows consoles default to cp1252
    s = get_settings()
    p = argparse.ArgumentParser(description="Ask a question to the document assistant.")
    p.add_argument("question")
    p.add_argument("--groups", nargs="*", default=[], help="the user's groups (public is always added)")
    p.add_argument("--mode", choices=["dense", "sparse", "hybrid"], default=s.retrieval_mode)
    p.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=s.rerank_enabled)
    p.add_argument("--no-generate", action="store_true", help="retrieval only, no LLM")
    p.add_argument("--collection", default=s.qdrant_collection)
    p.add_argument("--verbose", action="store_true", help="all candidates and full passage texts")
    args = p.parse_args()

    transformers_logging.disable_progress_bar()  # no "Loading weights" bars in the output
    start = time.perf_counter()
    pipeline = RagPipeline(s, load_reranker=args.rerank)
    load_seconds = time.perf_counter() - start
    try:
        result = pipeline.answer(args.question, args.groups, mode=args.mode, rerank=args.rerank,
                                 generate=not args.no_generate, collection=args.collection)
    except (RuntimeError, httpx.HTTPError) as exc:
        print(f"Error: {exc}")
        return 1

    stages = args.mode + (" + rerank" if args.rerank else "") + (" (no generation)" if args.no_generate else "")
    print(f"\nQuestion: {result.question}  (language: {result.language})")
    print(f"Groups: {', '.join(result.groups)} | {stages} | collection: {args.collection}")

    if args.verbose:
        print_candidates(result)
    if result.answer is None:
        print(f"\nPassages for the LLM ({len(result.passages)}):")
        for number, c in enumerate(result.passages, start=1):
            print_passage(number, c, args.verbose)
    else:
        print(f"\nAnswer:\n{result.answer}")
        if result.sources:
            print("\nSources:")
            for number, c in result.sources:
                print_passage(number, c, args.verbose)
        if args.verbose:
            uncited = [(n, c) for n, c in enumerate(result.passages, start=1) if not c.cited]
            if uncited:
                print("\nPassages given to the LLM but not cited:")
                for number, c in uncited:
                    print_passage(number, c, True)
    for warning in result.warnings:
        print(f"Warning: {warning}")
    print_timings(result, load_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
