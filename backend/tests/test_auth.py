"""Tests for app.auth: password hashes, login tokens, effective groups, who reads the audit log."""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.auth import (can_read_audit, create_token, effective_groups, hash_password, read_token,
                      verify_password)
from app.db import AuditEntry, User

SECRET = "a-test-secret-that-is-long-enough-for-hs256"


def user(role: str, groups: list[str]) -> User:
    return User(username=f"{role}-user", full_name="", role=role, groups=groups, password_hash="",
                created_at=datetime.now(timezone.utc))


def entry(user_groups: list[str]) -> AuditEntry:
    return AuditEntry(created_at=datetime.now(timezone.utc), username="someone", role="employee",
                      user_groups=user_groups, groups=user_groups + ["public"], question="q", language="en",
                      mode="hybrid", reranked=True, retrieved=[], passages=[], status="ok", seconds=1.0)


def test_password_hash_checks_the_right_password_only():
    stored = hash_password("correct horse")
    assert stored.startswith("scrypt$")
    assert "correct horse" not in stored
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)


def test_same_password_gives_different_hashes():
    assert hash_password("same") != hash_password("same")  # a random salt per hash


def test_malformed_stored_hash_is_refused():
    assert not verify_password("x", "")
    assert not verify_password("x", "md5$abc")


def test_token_round_trip():
    assert read_token(create_token("hr.officer", SECRET, 60), SECRET) == "hr.officer"


def test_expired_token_is_refused():
    token = create_token("hr.officer", SECRET, 60, now=datetime.now(timezone.utc) - timedelta(hours=2))
    with pytest.raises(jwt.ExpiredSignatureError):
        read_token(token, SECRET)


def test_token_signed_with_another_secret_is_refused():
    token = create_token("admin", "another-secret-that-is-also-long-enough", 60)
    with pytest.raises(jwt.InvalidSignatureError):
        read_token(token, SECRET)


def test_unsigned_token_is_refused():
    # "alg": "none" means no signature: anyone could write {"sub": "admin"}.
    token = jwt.encode({"sub": "admin", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
                       None, algorithm="none")
    with pytest.raises(jwt.InvalidTokenError):
        read_token(token, SECRET)


def test_effective_groups():
    all_groups = {"hr", "procurement", "it-security", "public"}
    assert effective_groups(user("employee", ["hr"]), all_groups, "public") == ["hr", "public"]
    assert effective_groups(user("manager", ["hr"]), all_groups, "public") == ["hr", "public"]
    assert effective_groups(user("admin", []), all_groups, "public") == sorted(all_groups)


def test_who_reads_which_audit_entries():
    hr_entry, procurement_entry, public_only_entry = entry(["hr"]), entry(["procurement"]), entry([])
    manager = user("manager", ["hr"])
    assert can_read_audit(manager, hr_entry, "public")
    assert not can_read_audit(manager, procurement_entry, "public")
    assert not can_read_audit(manager, public_only_entry, "public")  # sharing public does not count
    assert not can_read_audit(user("manager", ["public"]), hr_entry, "public")
    assert all(can_read_audit(user("admin", []), e, "public") for e in (hr_entry, procurement_entry, public_only_entry))
    assert not can_read_audit(user("employee", ["hr"]), hr_entry, "public")
