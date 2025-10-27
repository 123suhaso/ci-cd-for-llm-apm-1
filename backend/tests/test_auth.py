# backend/tests/test_auth.py

import os
import sys
from pathlib import Path

#    backend/tests/test_auth.py -> parents[1] == backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 2) Ensure import-time config checks don't fail during pytest collection.
#    Provide a harmless dummy DB URL because tests monkeypatch DB usage.
os.environ.setdefault("DATABASE_URL", "postgresql://ci:ci@127.0.0.1:5432/ci_db")

import hashlib
from datetime import datetime, timedelta, timezone
import uuid

import pytest
import asyncpg
from httpx import AsyncClient, ASGITransport

from app import app
import router.auth as auth_router  # patch get_connection, release_connection, send_otp_email_smtp


# Helper: simple fake async connection object
class FakeConn:
    def __init__(self):
        # simple in-memory "tables"
        self.users = {}
        self.password_otps = {}

    async def fetchrow(self, query, *args):
        q = (query or "").lower()
        # Signup insert: return inserted row
        if "insert into llm_apm.llm_users" in q:
            new_id = args[0]
            row = {
                "id": new_id,
                "name": args[1],
                "email": args[2],
                "username": args[3],
                "role": args[5],
                "is_active": args[6],
                "created_at": datetime.now(timezone.utc),
            }
            # check uniqueness by username or email
            for u in self.users.values():
                if u["username"] == row["username"] or u["email"] == row["email"]:
                    raise asyncpg.exceptions.UniqueViolationError("duplicate key")
            self.users[str(new_id)] = row
            return row

        # SELECT user by username for login
        if "from llm_apm.llm_users where username" in q:
            username = args[0]
            for uid, u in self.users.items():
                if u["username"] == username:
                    stored = {
                        "id": uid,
                        "username": u["username"],
                        "hashed_password": u.get("hashed_password"),
                        "role": u.get("role", "user"),
                        "is_active": u.get("is_active", True),
                    }
                    return stored
            return None

        # SELECT user by email for forgot-request-otp
        if "from llm_apm.llm_users where email" in q:
            email = args[0]
            for uid, u in self.users.items():
                if u["email"] == email:
                    return {"id": uid, "email": u["email"], "username": u.get("username")}
            return None

        # fetch the latest otp record for email
        if "from llm_apm.password_otps" in q and "expires_at" in q:
            email = args[0]
            now = args[1]
            found = None
            for rid, r in sorted(self.password_otps.items(), key=lambda x: x[1]["created_at"], reverse=True):
                if r["email"] == email and r["expires_at"] >= now:
                    found = r
                    break
            if not found:
                return None
            return {
                "id": found["id"],
                "user_id": found["user_id"],
                "otp_hash": found["otp_hash"],
                "expires_at": found["expires_at"],
            }

        return None

    async def execute(self, query, *args):
        q = (query or "").lower()
        # insert into password_otps
        if "insert into llm_apm.password_otps" in q:
            rec_id = args[0]
            user_id = args[1]
            email = args[2]
            otp_hash = args[3]
            expires_at = args[4]
            self.password_otps[rec_id] = {
                "id": rec_id,
                "user_id": user_id,
                "email": email,
                "otp_hash": otp_hash,
                "expires_at": expires_at,
                "created_at": datetime.now(timezone.utc),
            }
            return "OK"

        # update user hashed_password
        if "update llm_apm.llm_users set hashed_password" in q:
            new_hash = args[0]
            user_id = args[1]
            if str(user_id) in self.users:
                self.users[str(user_id)]["hashed_password"] = new_hash
                return "OK"
            return None

        # delete password_otps
        if "delete from llm_apm.password_otps" in q:
            user_id = args[0]
            email = args[1]
            to_del = [k for k, v in self.password_otps.items() if v["user_id"] == user_id or v["email"] == email]
            for k in to_del:
                del self.password_otps[k]
            return "OK"

        return "OK"


# Fixtures
@pytest.fixture
def fake_conn():
    return FakeConn()


@pytest.fixture(autouse=True)
def patch_connection(monkeypatch, fake_conn):
    """
    Monkeypatch auth_router.get_connection and release_connection so endpoints use the fake conn.
    We also patch send_otp_email_smtp to avoid real SMTP.
    """
    async def fake_get_connection():
        return fake_conn

    async def fake_release_connection(conn):
        return None

    def fake_send_otp_email_smtp(to_email: str, otp: str, username: str = None):
        fake_conn._last_sent_otp = {"email": to_email, "otp": otp, "username": username}

    monkeypatch.setattr(auth_router, "get_connection", fake_get_connection)
    monkeypatch.setattr(auth_router, "release_connection", fake_release_connection)
    monkeypatch.setattr(auth_router, "send_otp_email_smtp", fake_send_otp_email_smtp)
    yield


@pytest.mark.asyncio
async def test_signup_success(fake_conn):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "name": "Test User",
            "email": "test@example.com",
            "username": "testuser",
            "password": "Aa!strong1",
            "role": "user",
            "is_active": True,
        }
        r = await ac.post("/auth/users", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert body["message"] == "User created"
        assert body["user"]["username"] == "testuser"
        assert any(u["username"] == "testuser" for u in fake_conn.users.values())


@pytest.mark.asyncio
async def test_signup_duplicate(fake_conn):
    existing_id = str(uuid.uuid4())
    fake_conn.users[existing_id] = {
        "id": existing_id,
        "name": "Existing",
        "email": "dupe@example.com",
        "username": "dupeuser",
        "role": "user",
        "is_active": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        payload = {
            "name": "New One",
            "email": "dupe@example.com",
            "username": "dupeuser",
            "password": "Aa!strong1",
            "role": "user",
            "is_active": True,
        }
        r = await ac.post("/auth/users", json=payload)
        assert r.status_code in (400, 500)
        if r.status_code == 400:
            assert "already exists" in r.text.lower() or "duplicate" in r.text.lower()


@pytest.mark.asyncio
async def test_login_success(fake_conn):
    from router.auth import hash_password

    uid = str(uuid.uuid4())
    pw = "Aa!strong1"
    hashed = hash_password(pw)
    fake_conn.users[uid] = {
        "id": uid,
        "name": "LoginUser",
        "email": "login@example.com",
        "username": "loginuser",
        "hashed_password": hashed,
        "role": "user",
        "is_active": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/auth/login",
            data={"username": "loginuser", "password": pw},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 200
        j = r.json()
        assert "access_token" in j
        assert j["user"]["username"] == "loginuser"


@pytest.mark.asyncio
async def test_login_invalid_credentials(fake_conn):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post(
            "/auth/login",
            data={"username": "noone", "password": "badpass"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert r.status_code == 401


@pytest.mark.asyncio
async def test_forgot_request_otp_sends_or_silently_accepts(fake_conn):
    uid = str(uuid.uuid4())
    fake_conn.users[uid] = {
        "id": uid,
        "name": "OTPUser",
        "email": "otp@example.com",
        "username": "otpuser",
        "role": "user",
        "is_active": True,
    }

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        r = await ac.post("/auth/forgot-request-otp", json={"email": "otp@example.com"})
        assert r.status_code == 200
        body = r.json()
        assert any(v["email"] == "otp@example.com" for v in fake_conn.password_otps.values())
        assert hasattr(fake_conn, "_last_sent_otp")
        assert fake_conn._last_sent_otp["email"] == "otp@example.com"
