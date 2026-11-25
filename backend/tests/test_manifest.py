"""Tests for app.manifest: the real manifest is valid, and mistakes are rejected."""

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.manifest import DocumentEntry, Manifest, load_manifest

VALID = dict(id="hr-policy-en", title="HR policy", path="hr/policy.pdf", department="hr",
             language="en", allowed_groups=["hr"], source_url="https://example.org/policy.pdf")


def test_project_manifest_is_valid():
    manifest = load_manifest(get_settings().manifest_path)
    assert len(manifest.documents) >= 1
    assert all(d.allowed_groups for d in manifest.documents)


def test_valid_entry():
    entry = DocumentEntry(**VALID)
    assert entry.split is None and entry.manual_download is False and entry.exclude is None


@pytest.mark.parametrize("change", [
    {"path": "procurement/policy.pdf"},  # not in its department folder
    {"path": "hr/../secret.pdf"},  # escapes data/raw
    {"path": "hr/policy.exe"},  # unsupported type
    {"language": "French"},  # not a two-letter language code
    {"allowed_groups": []},  # nobody could read it
    {"allowed_groups": ["HR"]},  # group names are lowercase
    {"id": "Bad Id"},  # ids are slugs
    {"split": "dev"},  # only train/test
])
def test_invalid_entries_are_rejected(change):
    with pytest.raises(ValidationError):
        DocumentEntry(**{**VALID, **change})


def test_duplicate_ids_are_rejected():
    with pytest.raises(ValidationError):
        Manifest(documents=[VALID, {**VALID, "path": "hr/other.pdf"}])
