"""
Password hashing and JWT issuing for the serving API.

    JWT_SECRET         signing key - REQUIRED in any real deployment
    JWT_TTL_MINUTES    token lifetime                    (default 60)
    ALLOW_REGISTRATION "false" closes POST /auth/register (default true)

Passwords are hashed with `hashlib.scrypt`, a memory-hard KDF in the standard
library. Tokens are signed and verified with PyJWT.
"""
import hashlib
import hmac
import os
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import db

ALGORITHM = "HS256"
TTL_MINUTES = int(os.getenv("JWT_TTL_MINUTES", "60"))
ALLOW_REGISTRATION = os.getenv("ALLOW_REGISTRATION", "true").lower() != "false"

# n=2**14 puts a single hash around 50-100 ms: slow enough to make offline guessing
# expensive, fast enough not to delay a login.
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


def _load_secret() -> str:
    """Read JWT_SECRET, or generate an ephemeral one and warn.

    There is deliberately no hardcoded fallback: it would ship one forgeable secret
    to every deployment that forgot to set the variable. An ephemeral key fails safe
    instead - tokens stop working across a restart, which is visible and harmless.
    """
    secret = os.getenv("JWT_SECRET")
    if secret:
        return secret
    print("WARNING: JWT_SECRET is not set. Generating an ephemeral signing key - "
          "every issued token becomes invalid when this process restarts. Set "
          "JWT_SECRET to a random value to keep sessions across restarts.",
          flush=True)
    return secrets.token_urlsafe(48)


SECRET = _load_secret()


# ------------------------------------------------------------------ passwords
def hash_password(password: str) -> str:
    """Return 'salt$hash', both hex. A fresh salt per user, as scrypt requires."""
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode(), salt=salt, **_SCRYPT)
    return f"{salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, hash_hex = stored.split("$", 1)
        dk = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), **_SCRYPT)
    except (ValueError, TypeError):
        return False
    # compare_digest, not ==: an early-exit comparison leaks through timing.
    return hmac.compare_digest(dk.hex(), hash_hex)


# --------------------------------------------------------------------- tokens
def create_token(username: str) -> dict:
    expires = datetime.now(timezone.utc) + timedelta(minutes=TTL_MINUTES)
    token = jwt.encode({"sub": username, "exp": expires}, SECRET, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer",
            "expires_at": expires.isoformat(timespec="seconds"),
            "expires_in": TTL_MINUTES * 60}


_bearer = HTTPBearer(auto_error=False)


def current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)) -> str:
    """FastAPI dependency: verify the bearer token and return the username.

    401 with WWW-Authenticate, so a client can tell "not authenticated" from
    "not allowed".
    """
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"})

    if creds is None:
        raise unauthorized
    try:
        payload = jwt.decode(creds.credentials, SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise unauthorized

    username = payload.get("sub")
    if not username:
        raise unauthorized
    # A token can outlive the account it names.
    if db.get_user(username) is None:
        raise unauthorized
    return username
