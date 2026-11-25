"""Tests for app.langdetect."""

from app.langdetect import arabic_word_order, detect_language


def test_english():
    assert detect_language("The Borrower shall apply the Procurement Regulations to all contracts.") == "en"


def test_french():
    assert detect_language("L'emprunteur doit appliquer le règlement de passation des marchés.") == "fr"


def test_arabic():
    assert detect_language("يجب على المورد احترام الشروط والأحكام العامة لطلب الشراء.") == "ar"


def test_arabic_with_some_latin_words_is_arabic():
    assert detect_language("تطبق شروط طلب الشراء PO على جميع الموردين في البنك الدولي") == "ar"


def test_no_letters_is_unknown():
    assert detect_language("12 345 - 6.7") == "unknown"


def test_arabic_word_order():
    normal_text = "في هذا الطلب من المورد إلى المشتري"
    assert arabic_word_order(normal_text) == (4, 0)
    reversed_text = " ".join(w[::-1] for w in normal_text.split())
    normal, reversed_ = arabic_word_order(reversed_text)
    assert reversed_ > normal
