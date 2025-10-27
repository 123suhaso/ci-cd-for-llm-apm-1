import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import asyncpg
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# from llm_apm.storage.postgresql_async import AsyncPostgreSQLStorage

# storage = AsyncPostgreSQLStorage()


# async def get_connection():
#     await storage.init_pool()
#     return await storage._pool.acquire()


# async def release_connection(conn):
#     if storage._pool:
#         await storage._pool.release(conn)


router = APIRouter(tags=["auth"])

router = APIRouter(tags=["auth"])


@router.get("/ui-config")
async def ui_config():
    try:
        from config import AZURE_OPENAI_DEPLOYMENT
    except Exception:
        AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

    try:
        otp_minutes = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))
    except Exception:
        otp_minutes = 10

    return {
        "model_name": AZURE_OPENAI_DEPLOYMENT,
        "otp_expire_minutes": otp_minutes,
    }


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(p):
    return pwd_context.hash(p)


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


load_dotenv()
SECRET_KEY = os.getenv("LLM_APM_SECRET_KEY", "dev-secret-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

METRICS_JWT_SECRET = os.getenv("METRICS_JWT_SECRET", None)
METRICS_JWT_AUD = os.getenv("METRICS_JWT_AUD", "llm-apm")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def create_access_token(data: dict, expires_min: int = ACCESS_TOKEN_EXPIRE_MINUTES):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
    to_encode.update({"exp": int(expire.timestamp())})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


class UserCreate(BaseModel):
    name: str
    email: EmailStr
    username: str
    password: str
    role: str = "user"
    is_active: bool = True


def _verify_service_token(token: str) -> Optional[dict]:
    secret = METRICS_JWT_SECRET or SECRET_KEY
    try:
        if token.count(".") == 2:
            payload = jwt.decode(
                token, secret, algorithms=[ALGORITHM], audience=METRICS_JWT_AUD
            )
            if (
                payload.get("role") == "metrics_scraper"
                or payload.get("sub") == "prometheus-scraper"
            ):
                return {
                    "id": "prometheus-service",
                    "username": "prometheus",
                    "role": payload.get("role", "metrics_scraper"),
                    "service": True,
                }
            return None
    except JWTError:
        pass
    if token == secret:
        return {
            "id": "prometheus-service",
            "username": "prometheus",
            "role": "metrics_scraper",
            "service": True,
        }
    return None


def _verify_user_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None
    user_id = payload.get("id") or payload.get("sub") or payload.get("user_id")
    username = payload.get("sub") or payload.get("username")
    role = payload.get("role")
    if not user_id or not username:
        return None
    return {"id": user_id, "username": username, "role": role, "claims": payload}


def get_current_user(token: str = Depends(oauth2_scheme)):
    if not token:
        raise HTTPException(status_code=401, detail="Missing Authorization token")
    svc = _verify_service_token(token)
    if svc:
        return svc
    user = _verify_user_token(token)
    if user:
        return user
    raise HTTPException(status_code=401, detail="Invalid or expired token")


def optional_get_current_user(token: Optional[str] = Depends(oauth2_scheme)):
    if not token:
        return None
    try:
        svc = _verify_service_token(token)
        if svc:
            return svc
        user = _verify_user_token(token)
        if user:
            return user
        return None
    except Exception:
        return None


PASSWORD_REGEX = re.compile(r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$")


def is_password_valid(pw: str) -> bool:
    if not pw:
        return False
    return bool(PASSWORD_REGEX.match(pw))


def is_email_valid(email: str) -> bool:
    if not email or "@" not in email:
        return False
    return email.lower().endswith(".com")


@router.post("/users")
async def signup(user: UserCreate):
    if not is_email_valid(user.email):
        raise HTTPException(
            status_code=400, detail="Email must contain '@' and end with '.com'"
        )
    if not is_password_valid(user.password):
        raise HTTPException(
            status_code=400,
            detail="Password policy: minimum 8 chars, at least 1 uppercase, 1 lowercase, 1 number and 1 special character",
        )
    conn = None
    try:
        conn = await get_connection()
        new_id = str(uuid.uuid4())
        try:
            row = await conn.fetchrow(
                """
                INSERT INTO llm_apm.llm_users
                 (id, name, email, username, hashed_password, role, is_active, created_at)
                VALUES ($1,$2,$3,$4,$5,$6,$7, now())
                RETURNING id::text as id, name, email, username, role, is_active, created_at
                """,
                new_id,
                user.name,
                user.email,
                user.username,
                hash_password(user.password),
                user.role,
                user.is_active,
            )
            return {"message": "User created", "user": dict(row) if row else None}
        except asyncpg.exceptions.UniqueViolationError:
            raise HTTPException(
                status_code=400, detail="username or email already exists"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await release_connection(conn)


@router.post("/login")
async def login(form: OAuth2PasswordRequestForm = Depends()):
    conn = None
    try:
        conn = await get_connection()
        row = await conn.fetchrow(
            "SELECT id::text as id, username, hashed_password, role, is_active FROM llm_apm.llm_users WHERE username = $1",
            form.username,
        )
        if not row or not verify_password(form.password, row["hashed_password"]):
            raise HTTPException(status_code=401, detail="Invalid username or password")
        if not row.get("is_active", True):
            raise HTTPException(status_code=403, detail="Inactive user")
        token = create_access_token(
            {"sub": row["username"], "role": row["role"], "id": str(row["id"])}
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": str(row["id"]),
                "username": row["username"],
                "role": row["role"],
            },
        }
    finally:
        if conn:
            await release_connection(conn)


@router.get("/me")
async def me(current_user: dict = Depends(get_current_user)):
    return {"user": current_user}


import hashlib
import random
import smtplib
import string
from email.message import EmailMessage

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL", SMTP_USER or "noreply@example.com")

OTP_LENGTH = int(os.getenv("OTP_LENGTH", "6"))
OTP_EXPIRE_MINUTES = int(os.getenv("OTP_EXPIRE_MINUTES", "10"))


def _generate_numeric_otp(length: int = OTP_LENGTH) -> str:
    return "".join(random.choices(string.digits, k=length))


def _hash_otp(otp: str) -> str:
    return hashlib.sha256(otp.encode("utf-8")).hexdigest()


def send_otp_email_smtp(to_email: str, otp: str, username: Optional[str] = None):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        raise RuntimeError("SMTP config not set (SMTP_HOST/SMTP_USER/SMTP_PASS)")
    msg = EmailMessage()
    msg["Subject"] = "Your password reset OTP"
    msg["From"] = FROM_EMAIL
    msg["To"] = to_email
    body = f"Hello{f' {username}' if username else ''},\n\n"
    body += f"Use the following OTP to reset your password. It expires in {OTP_EXPIRE_MINUTES} minutes.\n\n"
    body += f"OTP: {otp}\n\n"
    body += "If you did not request this, ignore this email.\n\nThanks."
    msg.set_content(body)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)


class RequestOtpRequest(BaseModel):
    email: EmailStr


@router.post("/forgot-request-otp")
async def forgot_request_otp(payload: RequestOtpRequest):
    conn = None
    try:
        conn = await get_connection()
        row = await conn.fetchrow(
            "SELECT id::text as id, email, username FROM llm_apm.llm_users WHERE email = $1",
            payload.email,
        )
        if not row:
            return {
                "message": "If an account exists for that email, an OTP has been sent."
            }
        user_id = row["id"]
        username = row["username"]
        otp = _generate_numeric_otp()
        otp_hash = _hash_otp(otp)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=OTP_EXPIRE_MINUTES)
        await conn.execute(
            """
            INSERT INTO llm_apm.password_otps (id, user_id, email, otp_hash, expires_at, created_at)
            VALUES ($1, $2, $3, $4, $5, now())
            """,
            str(uuid.uuid4()),
            user_id,
            payload.email,
            otp_hash,
            expires_at,
        )
        try:
            send_otp_email_smtp(payload.email, otp, username=username)
        except Exception as e:
            print("OTP email send failed:", e)
        return {"message": "If an account exists for that email, an OTP has been sent."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await release_connection(conn)


class VerifyOtpAndResetRequest(BaseModel):
    email: EmailStr
    otp: str
    new_password: str


@router.post("/forgot-verify-otp")
async def forgot_verify_otp(payload: VerifyOtpAndResetRequest):
    if not is_password_valid(payload.new_password):
        raise HTTPException(
            status_code=400,
            detail="Password policy: minimum 8 chars, at least 1 uppercase, 1 lowercase, 1 number and 1 special character",
        )
    conn = None
    try:
        conn = await get_connection()
        now = datetime.now(timezone.utc)
        record = await conn.fetchrow(
            """
            SELECT id::text as id, user_id::text as user_id, otp_hash, expires_at
            FROM llm_apm.password_otps
            WHERE email = $1 AND expires_at >= $2
            ORDER BY created_at DESC
            LIMIT 1
            """,
            payload.email,
            now,
        )
        if not record:
            raise HTTPException(status_code=400, detail="Invalid or expired OTP")
        provided_hash = _hash_otp(payload.otp)
        if provided_hash != record["otp_hash"]:
            raise HTTPException(status_code=400, detail="Invalid OTP")
        new_hashed = hash_password(payload.new_password)
        await conn.execute(
            "UPDATE llm_apm.llm_users SET hashed_password = $1 WHERE id = $2",
            new_hashed,
            record["user_id"],
        )
        await conn.execute(
            "DELETE FROM llm_apm.password_otps WHERE user_id = $1 OR email = $2",
            record["user_id"],
            payload.email,
        )
        return {"message": "Password has been reset successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            await release_connection(conn)
