import secrets
import smtplib
import uuid
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import bcrypt
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.user import User, EmailVerification


# ── Password ──────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


# ── JWT ───────────────────────────────────────────────────────────────────────

def create_access_token(user_id: uuid.UUID) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": str(user_id), "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Returns user_id string or None if invalid."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ── User queries ──────────────────────────────────────────────────────────────

async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def create_user(db: AsyncSession, email: str, password: str) -> User:
    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    await db.flush()  # get the id without committing
    return user


# ── Email verification ────────────────────────────────────────────────────────

async def create_verification_token(db: AsyncSession, user: User) -> str:
    token = secrets.token_urlsafe(32)
    verification = EmailVerification(
        user_id=user.id,
        token=token,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )
    db.add(verification)
    await db.flush()
    return token


async def verify_email_token(db: AsyncSession, token: str) -> User | None:
    result = await db.execute(
        select(EmailVerification).where(
            EmailVerification.token == token,
            EmailVerification.used == False,  # noqa: E712
            EmailVerification.expires_at > datetime.now(timezone.utc),
        )
    )
    verification = result.scalar_one_or_none()
    if not verification:
        return None

    verification.used = True
    user = await get_user_by_id(db, verification.user_id)
    if user:
        user.is_verified = True
    return user


# ── Email dispatch ────────────────────────────────────────────────────────────

def send_verification_email(email: str, token: str) -> None:
    """
    Send email verification link.
    Falls back to stdout if SMTP is not configured (development mode).
    """
    verification_url = f"{settings.FRONTEND_URL}/verify-email?token={token}"

    if not settings.SMTP_HOST:
        # Development: print to stdout
        print(f"\n[DEV] Email verification link for {email}:")
        print(f"  {verification_url}\n")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Verify your {settings.APP_NAME} account"
    msg["From"] = settings.SMTP_FROM
    msg["To"] = email

    html = f"""
    <html><body>
      <h2>Welcome to {settings.APP_NAME}</h2>
      <p>Click the link below to verify your email address:</p>
      <p><a href="{verification_url}">{verification_url}</a></p>
      <p>This link expires in 24 hours.</p>
    </body></html>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
            server.starttls()
            server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
            server.sendmail(settings.SMTP_FROM, email, msg.as_string())
    except Exception as exc:
        # Non-fatal: user can request a new link
        print(f"[WARN] Failed to send verification email to {email}: {exc}")
