"""
auth.py — JWT-based authentication with bcrypt password hashing.

Provides:
  - Password hashing & verification (bcrypt)
  - JWT token creation & decoding (HS256, HTTP-only cookie)
  - FastAPI dependency: get_current_user (extracts user from cookie)
  - Role-based access helpers

Install requirements:
  pip install bcrypt PyJWT
"""

import bcrypt
import jwt
from datetime import datetime, timedelta, timezone
from fastapi import Request, HTTPException, status

# ═══════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════

# IMPORTANT: Change this to a strong random secret in production.
# Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY = "qp-engine-secret-change-me-in-production-2025"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 8
JWT_COOKIE_NAME = "qp_access_token"


# ═══════════════════════════════════════════
# PASSWORD HASHING (bcrypt)
# ═══════════════════════════════════════════

def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against a bcrypt hash."""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# ═══════════════════════════════════════════
# JWT TOKEN
# ═══════════════════════════════════════════

def create_access_token(user_id: str, email: str, role: str, name: str) -> str:
    """Create a signed JWT with user claims."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "name": name,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRATION_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """
    Decode and verify a JWT token.
    Returns the payload dict on success.
    Raises HTTPException on failure.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired. Please login again.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token.",
        )


# ═══════════════════════════════════════════
# FASTAPI DEPENDENCY — get_current_user
# ═══════════════════════════════════════════

async def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency that extracts and validates the JWT from
    the HTTP-only cookie. Returns the decoded user payload.

    Usage in route:
        @app.get("/protected")
        async def protected_route(user: dict = Depends(get_current_user)):
            return {"email": user["email"]}
    """
    token = request.cookies.get(JWT_COOKIE_NAME)

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated. No access token cookie found.",
        )

    return decode_access_token(token)


# ═══════════════════════════════════════════
# ROLE-BASED ACCESS HELPERS
# ═══════════════════════════════════════════

def require_role(user: dict, allowed_roles: list[str]):
    """
    Check if the user has one of the allowed roles.
    Raises 403 if not.

    Usage:
        require_role(user, ["admin"])
    """
    if user.get("role") not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied. Required role: {', '.join(allowed_roles)}",
        )