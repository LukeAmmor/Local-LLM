"""Authentication and authorisation: passwords, login tokens (JWT), roles, groups.

Plain functions, no web code: app.api uses them, and they are tested alone.

Passwords: never stored, only a scrypt hash. scrypt (in Python's hashlib)
is slow and needs memory on purpose (here about 50 ms and 16 MiB per hash),
so trying millions of passwords against a stolen database is expensive. A
random salt per user makes the same password give a different hash for each
user. The stored text holds everything needed to check a password later:
"scrypt$n$r$p$salt$hash" (salt and hash in hex).

Login tokens: a JWT (JSON Web Token) signed with HS256, an HMAC-SHA256 of
the token content with the secret setting jwt_secret. It holds only the
username ("sub") and an expiry time ("exp"). Whoever changes one character
breaks the signature, and only the server knows the secret, so the server
can trust a token without storing it. The role and groups are NOT in the
token: the API reads them from the database at every request, so a change
to a user applies at once, not when their token expires. The accepted
algorithm is fixed (algorithms=["HS256"]): a token that claims another one,
such as "none" (no signature), is refused.

Roles (one per user):
  employee  asks questions; reads the documents of their groups + public.
  manager   same, and reads the audit log of their own departments: entries
            of users who share one of the manager's groups (public excluded).
  admin     reads every group, sees every audit entry, lists users and documents.
The access filter is never switched off: an admin's groups are simply all the
groups used by the documents.
"""

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

import jwt

from app.db import AuditEntry, User
from app.retrieval import user_groups

ROLES = ("employee", "manager", "admin")
JWT_ALGORITHM = "HS256"
MIN_SECRET_LENGTH = 32  # characters; HS256 needs a secret of at least 256 bits to be safe
SCRYPT_N, SCRYPT_R, SCRYPT_P = 2**14, 8, 1  # cost: memory = 128 * N * r bytes = 16 MiB


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        name, n, r, p, salt, digest = stored.split("$")
    except ValueError:
        return False
    if name != "scrypt":
        return False
    candidate = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p))
    # compare_digest takes the same time wherever the first difference is, so
    # the response time tells nothing about how much of the hash was right.
    return hmac.compare_digest(candidate, bytes.fromhex(digest))


# Checked when the username does not exist, so that an unknown username takes
# as long as a wrong password: the response time does not reveal which
# usernames exist.
DUMMY_HASH = hash_password(secrets.token_hex(16))


def create_token(username: str, secret: str, expire_minutes: int, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    claims = {"sub": username, "iat": now, "exp": now + timedelta(minutes=expire_minutes)}
    return jwt.encode(claims, secret, algorithm=JWT_ALGORITHM)


def read_token(token: str, secret: str) -> str:
    """The username of a valid token. Raises jwt.InvalidTokenError: bad signature, expired, malformed..."""
    claims = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM], options={"require": ["sub", "exp"]})
    return claims["sub"]


def effective_groups(user: User, all_groups: set[str], public_group: str) -> list[str]:
    """The groups given to the access filter: the user's groups + public; for an admin, every group."""
    if user.role == "admin":
        return sorted(all_groups | {public_group})
    return user_groups(user.groups, public_group)


def can_read_audit(viewer: User, entry: AuditEntry, public_group: str) -> bool:
    """Admin: every entry. Manager: entries of users who share one of the manager's departments."""
    if viewer.role == "admin":
        return True
    if viewer.role == "manager":
        departments = set(viewer.groups) - {public_group}  # everyone shares public: it does not count
        return bool(departments & (set(entry.user_groups) - {public_group}))
    return False
