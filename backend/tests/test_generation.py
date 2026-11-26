"""Tests for app.generation: citation checking, "I don't know" detection, prompt contents.

No LLM is called: these are the parts of generation that are plain code.
"""

from app.generation import IDK, build_messages, check_citations, is_refusal, location

PDF = {"title": "ILO Circular 543", "page_start": 2, "page_end": 2, "section": "Procedures", "text": "Report it."}
DOCX = {"title": "Règlement", "page_start": None, "page_end": None, "section": "Plan de Passation", "text": "Le plan."}


def test_valid_citations_are_kept_and_invalid_ones_removed():
    text, cited, invalid = check_citations("A [1]. B [1][3]. C [2, 9]. D [7].", n_passages=5)
    assert text == "A [1]. B [1][3]. C [2]. D."
    assert cited == {1, 2, 3}
    assert invalid == [9, 7]


def test_zero_is_not_a_passage_number():
    text, cited, invalid = check_citations("Nothing here [0].", n_passages=5)
    assert (text, cited, invalid) == ("Nothing here.", set(), [0])


def test_refusal_is_recognised():
    assert is_refusal(IDK["en"], set())
    # The French sentence with a narrow no-break space before the colon (French typography).
    assert is_refusal("Je ne sais pas : les documents auxquels vous avez accès ne répondent pas à cette question.",
                      set())
    assert is_refusal("I don't know.", set())  # starts like the sentence and cites nothing


def test_an_answer_is_not_a_refusal():
    assert not is_refusal("Report it to your supervisor [1].", {1})
    assert not is_refusal("I don't know the exact date, but the policy applies to all staff [1].", {1})


def test_location_pages_and_sections():
    assert location(PDF) == "p. 2"
    assert location({**PDF, "page_end": 3}) == "pp. 2-3"
    assert location(DOCX) == "section: Plan de Passation"
    assert location({**DOCX, "section": ""}) == ""


def test_prompt_numbers_passages_and_sets_the_language():
    system, user = build_messages("Que faire en cas de harcèlement ?", [PDF, DOCX], "fr")
    assert system["role"] == "system" and user["role"] == "user"
    assert "Write your answer in French" in system["content"]
    assert IDK["fr"] in system["content"]
    assert "[1] ILO Circular 543, p. 2\nReport it." in user["content"]
    assert "[2] Règlement, section: Plan de Passation\nLe plan." in user["content"]
    assert user["content"].endswith("Question: Que faire en cas de harcèlement ?")


def test_unknown_language_offers_both_sentences():
    system, _ = build_messages("Mot de passe ?", [PDF], "unknown")
    assert "in the language of the question" in system["content"]
    assert IDK["en"] in system["content"] and IDK["fr"] in system["content"]
