"""Tests for app.langdetect."""

from app.langdetect import detect_language


def test_english():
    assert detect_language("The Borrower shall apply the Procurement Regulations to all contracts.") == "en"


def test_french():
    assert detect_language("L'emprunteur doit appliquer le règlement de passation des marchés.") == "fr"


def test_english_with_a_french_name_is_english():
    assert detect_language("The Agence nationale de la sécurité publishes the guide for all the teams.") == "en"


def test_other_script_is_other():
    assert detect_language("Это руководство по информационной безопасности.") == "other"  # Russian
    assert detect_language("يجب على المورد احترام الشروط PO العامة") == "other"  # Arabic with a Latin word


def test_no_letters_is_unknown():
    assert detect_language("12 345 - 6.7") == "unknown"
