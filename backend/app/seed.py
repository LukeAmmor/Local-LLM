"""Create (or update) the demo users of the video.

Usage:  uv run python -m app.seed

All demo users share one password, the setting seed_password: set
RAG_SEED_PASSWORD in .env (gitignored). The repository is public, so the
password is never written in the code; the script refuses to run without it.

  username             role      groups        reads
  procurement.officer  employee  procurement   procurement + public
  hr.officer           employee  hr            hr + public
  hr.manager           manager   hr            hr + public, and the audit log of hr users
  security.officer     employee  it-security   it-security + public
  admin                admin     (none)        every group, the whole audit log, users, documents

Running it again is safe: the demo users get their name, role, groups and
password back (so changing RAG_SEED_PASSWORD and re-running changes their
password); other users are not touched. The groups are checked against the
manifest: a typo there would silently give a user no documents.
"""

import sys

from sqlmodel import Session

from app.auth import ROLES, hash_password
from app.config import get_settings
from app.db import get_engine, save_user
from app.manifest import load_manifest

MIN_PASSWORD_LENGTH = 8

DEMO_USERS = [  # username, full name, role, groups
    ("procurement.officer", "Procurement Officer", "employee", ["procurement"]),
    ("hr.officer", "HR Officer", "employee", ["hr"]),
    ("hr.manager", "HR Manager", "manager", ["hr"]),
    ("security.officer", "IT Security Officer", "employee", ["it-security"]),
    ("admin", "Administrator", "admin", []),
]


def main() -> int:
    s = get_settings()
    password = s.seed_password.get_secret_value()
    if len(password) < MIN_PASSWORD_LENGTH:
        print(f"Error: set RAG_SEED_PASSWORD in .env (at least {MIN_PASSWORD_LENGTH} characters): "
              "the password of the demo users.")
        return 1

    manifest = load_manifest(s.manifest_path)
    known_groups = {g for d in manifest.documents if not d.exclude for g in d.allowed_groups}
    for username, _, role, groups in DEMO_USERS:
        unknown = set(groups) - known_groups
        if role not in ROLES or unknown:
            print(f"Error: {username}: role {role!r} or groups {sorted(unknown)} unknown "
                  f"(groups of the manifest: {', '.join(sorted(known_groups))})")
            return 1

    with Session(get_engine(s.db_path)) as session:
        for username, full_name, role, groups in DEMO_USERS:
            save_user(session, username, full_name, role, groups, hash_password(password))
            print(f"  {username:<20} {role:<9} {', '.join(groups) or '(all groups)'}")
    print(f"{len(DEMO_USERS)} demo users ready in {s.db_path} (password: RAG_SEED_PASSWORD).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
