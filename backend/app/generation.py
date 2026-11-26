"""Generate the answer with the local LLM (qwen3:8b served by Ollama) and check its citations.

What:
  - `build_messages(question, passages, language)`: the prompt. A system
    message with the rules, and a user message with the numbered passages
    ("[1] Title, p. 12" + chunk text) followed by the question.
  - `chat(...)`: one request to Ollama's /api/chat over plain HTTP (httpx).
  - `check_citations(answer, n_passages)`: finds the [n] markers in the
    answer, removes the invalid ones (a number that is not a passage), and
    returns which passages are cited.
  - `is_refusal(answer, cited)`: is the answer the "I don't know" sentence?

The rules given to the LLM:
  1. use only the passages, no outside knowledge;
  2. cite the passage number after each sentence, like [2] or [1][3];
  3. if the passages do not contain the answer, reply with one fixed
     "I don't know" sentence, so the program (and later the UI and the
     evaluation) can tell a refusal from an answer. It says "the documents
     you have access to": it must not hint that a document the user may
     not read holds the answer;
  4. answer in the language of the question, which app.langdetect detects
     ("Write your answer in French"). The passages are often in the other
     language, and a small model tends to answer in the passages' language
     unless told. Short questions can be "unknown": the rule then says "in
     the language of the question".

Ollama options:
  - "think": false: qwen3 would otherwise write a long reasoning first.
  - "truncate": false: a prompt longer than num_ctx is refused with an error.
    By default Ollama cuts it silently (measured: a 5,126-token prompt was
    cut to 2,050 tokens with num_ctx 4096, without any error).
  - temperature 0 (setting): the same question gives the same answer.
"""

import re
from dataclasses import dataclass

import httpx

# The fixed "I don't know" sentences, one per supported language.
IDK = {
    "en": "I don't know: the documents you have access to do not answer this question.",
    "fr": "Je ne sais pas : les documents auxquels vous avez accès ne répondent pas à cette question.",
}
IDK_OPENINGS = ("i don't know", "je ne sais pas")
LANGUAGE_NAMES = {"en": "English", "fr": "French"}

SYSTEM_PROMPT = """You answer questions from employees using only the numbered passages from internal documents given with the question.

Rules:
1. Use only the information in the passages. Do not add facts from your own knowledge.
2. After each sentence, cite the passage it comes from with its number in square brackets, like [2]. To cite several passages, write [1][3]. Only cite numbers from the list of passages.
3. {idk_rule}
4. {language_rule}
5. Be concise: a short paragraph or a short list."""

# A citation marker, with the spaces before it: "[2]" or "[1, 3]". "[1][3]" is two markers.
CITATION = re.compile(r"(\s*)\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass
class LLMReply:
    text: str
    prompt_tokens: int  # length of the prompt, in qwen3 tokens
    answer_tokens: int
    prompt_seconds: float  # time to read the prompt ("prefill")
    answer_seconds: float  # time to write the answer
    load_seconds: float  # time to load the model (0 when it was already loaded)
    done_reason: str  # "stop" (finished) or "length" (cut at num_predict)


def idk_sentence(language: str) -> str:
    return IDK.get(language, IDK["en"])


def location(payload: dict) -> str:
    """Where a chunk is in its document: "p. 12", "pp. 12-13", or the section (DOCX: no pages)."""
    start, end = payload.get("page_start"), payload.get("page_end")
    if start is not None:
        return f"p. {start}" if start == end else f"pp. {start}-{end}"
    return f"section: {payload['section']}" if payload.get("section") else ""


def passage_label(number: int, payload: dict) -> str:
    where = location(payload)
    return f"[{number}] {payload['title']}" + (f", {where}" if where else "")


def build_messages(question: str, passages: list[dict], language: str) -> list[dict]:
    """The chat messages for Ollama. `passages` are chunk payloads, numbered [1], [2]... in this order."""
    if language in IDK:
        idk_rule = ("If the passages do not contain the answer, reply with exactly this sentence "
                    f"and nothing else: \"{IDK[language]}\"")
        language_rule = (f"Write your answer in {LANGUAGE_NAMES[language]}, "
                         "even when the passages are in another language.")
    else:
        idk_rule = ("If the passages do not contain the answer, reply with exactly one of these sentences, "
                    "in the language of the question, and nothing else: "
                    + " or ".join(f"\"{s}\"" for s in IDK.values()))
        language_rule = "Write your answer in the language of the question, even when the passages are in another language."
    system = SYSTEM_PROMPT.format(idk_rule=idk_rule, language_rule=language_rule)
    context = "\n\n".join(f"{passage_label(n, p)}\n{p['text']}" for n, p in enumerate(passages, start=1))
    user = f"Passages:\n\n{context}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def chat(ollama_url: str, model: str, messages: list[dict], num_ctx: int, num_predict: int,
         temperature: float, keep_alive: str, timeout: float) -> LLMReply:
    response = httpx.post(f"{ollama_url}/api/chat", timeout=timeout, json={
        "model": model,
        "messages": messages,
        "think": False,  # qwen3: no reasoning text before the answer
        "stream": False,  # the whole answer in one response
        "truncate": False,  # refuse a prompt longer than num_ctx instead of cutting it silently
        "keep_alive": keep_alive,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature},
    })
    if response.status_code != 200:
        raise RuntimeError(f"Ollama refused the request (HTTP {response.status_code}): "
                           f"{response.json().get('error', response.text)}")
    data = response.json()
    ns = 1e9  # Ollama gives durations in nanoseconds
    return LLMReply(
        text=data["message"]["content"].strip(),
        prompt_tokens=data.get("prompt_eval_count", 0),
        answer_tokens=data.get("eval_count", 0),
        prompt_seconds=data.get("prompt_eval_duration", 0) / ns,
        answer_seconds=data.get("eval_duration", 0) / ns,
        load_seconds=data.get("load_duration", 0) / ns,
        done_reason=data.get("done_reason", ""),
    )


def check_citations(answer: str, n_passages: int) -> tuple[str, set[int], list[int]]:
    """Return (answer without invalid markers, cited passage numbers, invalid numbers found).

    A valid number is between 1 and n_passages. "[2, 9]" with 5 passages becomes "[2]";
    "[9]" alone is removed with the spaces before it.
    """
    cited: set[int] = set()
    invalid: list[int] = []

    def keep_valid(match: re.Match) -> str:
        spaces, numbers = match.group(1), [int(n) for n in match.group(2).split(",")]
        valid = [n for n in numbers if 1 <= n <= n_passages]
        invalid.extend(n for n in numbers if not 1 <= n <= n_passages)
        cited.update(valid)
        return f"{spaces}[{', '.join(map(str, valid))}]" if valid else ""

    return CITATION.sub(keep_valid, answer), cited, invalid


def normalize(text: str) -> str:
    """Lowercase, typographic apostrophes and spaces made plain, for comparing sentences."""
    return " ".join(text.replace("’", "'").replace(" ", " ").replace(" ", " ").lower().split())


def is_refusal(answer: str, cited: set[int]) -> bool:
    """The answer is "I don't know": the fixed sentence, or an answer that starts like it and cites nothing."""
    text = normalize(answer)
    if any(normalize(sentence).rstrip(".") in text for sentence in IDK.values()):
        return True
    return text.startswith(IDK_OPENINGS) and not cited
