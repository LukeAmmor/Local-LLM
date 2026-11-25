"""Load and validate data/manifest.yaml, the list of documents.

What: turns the YAML file into a list of `DocumentEntry` objects, and refuses
a manifest with mistakes in it: a missing field, a malformed language code,
a duplicated id, a path outside the department folder, no access group...

Why: the manifest decides who may read each document (`allowed_groups`).
A typo there is an access-control bug, so we want it to fail loudly at load
time, not silently at question time.
"""

from pathlib import Path, PurePosixPath
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
SLUG = r"^[a-z0-9][a-z0-9-]*$"  # lowercase letters, digits and dashes


class DocumentEntry(BaseModel):
    id: str = Field(pattern=SLUG)
    title: str = Field(min_length=1)
    path: str  # relative to data/raw/, always "<department>/<file>"
    department: str = Field(pattern=SLUG)
    # Declared language, as a two-letter code ("en", "fr"...). Which languages
    # are indexed is a setting (supported_languages), checked by app.ingest.
    language: str = Field(pattern=r"^[a-z]{2}$")
    allowed_groups: list[str] = Field(min_length=1)
    source_url: str
    manual_download: bool = False
    exclude: str | None = None  # reason the document is kept out of the index (checked, never indexed)
    split: Literal["train", "test"] | None = None

    @field_validator("allowed_groups")
    @classmethod
    def groups_are_slugs(cls, groups: list[str]) -> list[str]:
        for group in groups:
            if not group or not group.replace("-", "").isalnum() or group != group.lower():
                raise ValueError(f"invalid group name {group!r} (use lowercase letters, digits, dashes)")
        return groups

    @model_validator(mode="after")
    def path_is_inside_department(self) -> "DocumentEntry":
        p = PurePosixPath(self.path)
        if p.is_absolute() or ".." in p.parts:
            raise ValueError(f"{self.id}: path must be relative and stay inside data/raw/")
        if len(p.parts) != 2 or p.parts[0] != self.department:
            raise ValueError(f"{self.id}: path must be '{self.department}/<file>', got {self.path!r}")
        if p.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"{self.id}: unsupported file type {p.suffix!r}")
        return self

    def file_path(self, raw_dir: Path) -> Path:
        """Absolute location of the file on disk."""
        return raw_dir / self.path


class Manifest(BaseModel):
    documents: list[DocumentEntry]

    @model_validator(mode="after")
    def ids_and_paths_are_unique(self) -> "Manifest":
        for field in ("id", "path"):
            values = [getattr(d, field) for d in self.documents]
            duplicates = {v for v in values if values.count(v) > 1}
            if duplicates:
                raise ValueError(f"duplicated {field}: {sorted(duplicates)}")
        return self

    def get(self, doc_id: str) -> DocumentEntry:
        for d in self.documents:
            if d.id == doc_id:
                return d
        raise KeyError(f"no document with id {doc_id!r} in the manifest")


def load_manifest(path: Path) -> Manifest:
    """Read and validate the manifest file (raises pydantic.ValidationError on mistakes)."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest.model_validate(data)
