# import asyncio
# import logging
# import os
# import sys
# from contextlib import asynccontextmanager
# from typing import AsyncGenerator, Optional

# import asyncpg

# logger = logging.getLogger("db")
# if not logger.handlers:
#     ch = logging.StreamHandler(sys.stdout)
#     ch.setFormatter(
#         logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
#     )
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)

# DEFAULT_DATABASE_URL = os.getenv("DATABASE_URL", None)

# _pool: Optional[asyncpg.Pool] = None
# _pool_lock: Optional[asyncio.Lock] = None


# async def _get_lock() -> asyncio.Lock:
#     global _pool_lock
#     if _pool_lock is None:
#         _pool_lock = asyncio.Lock()
#     return _pool_lock


# async def init_db_pool(
#     database_url: Optional[str] = None, min_size: int = 1, max_size: int = 10
# ) -> None:
#     """
#     Initialize the global asyncpg pool if not already initialized.
#     Safe to call concurrently.
#     """
#     global _pool
#     url = database_url or os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL
#     if not url:
#         raise RuntimeError("DATABASE_URL environment variable is not set.")

#     # fast-path
#     if _pool is not None:
#         return

#     lock = await _get_lock()
#     async with lock:
#         if _pool is None:
#             logger.info(
#                 "Creating asyncpg pool -> %s", url.split("@")[-1] if "@" in url else url
#             )
#             _pool = await asyncpg.create_pool(url, min_size=min_size, max_size=max_size)
#             logger.info("Database pool initialized.")


# async def get_connection() -> asyncpg.Connection:
#     """
#     Acquire a raw connection from the pool. Caller must release it.
#     Prefer using connection_ctx() below instead.
#     """
#     if _pool is None:
#         await init_db_pool()
#     return await _pool.acquire()


# async def release_connection(conn: asyncpg.Connection) -> None:
#     """
#     Safely release a connection back to the pool.
#     """
#     if conn is None:
#         return
#     if _pool is None:
#         logger.warning("release_connection: pool missing; closing connection directly.")
#         try:
#             await conn.close()
#         except Exception:
#             logger.exception("Failed to close connection when pool missing.")
#         return

#     try:
#         await _pool.release(conn)
#     except Exception:
#         logger.exception("Failed to release connection; attempting to close.")
#         try:
#             await conn.close()
#         except Exception:
#             logger.exception("Failed to close connection after failed release.")


# async def close_pool() -> None:
#     """
#     Close and clear the global pool.
#     """
#     global _pool
#     if _pool:
#         try:
#             await _pool.close()
#             logger.info("Database pool closed.")
#         except Exception:
#             logger.exception("Error while closing DB pool.")
#         finally:
#             _pool = None


# async def health_check() -> bool:
#     """
#     Simple check that the database responds to a query.
#     """
#     try:
#         if _pool is None:
#             await init_db_pool()
#         async with _pool.acquire() as conn:
#             row = await conn.fetchrow("SELECT 1 AS ok")
#             return bool(row and row["ok"] == 1)
#     except Exception:
#         logger.exception("DB health check failed")
#         return False


# @asynccontextmanager
# async def connection_ctx() -> AsyncGenerator[asyncpg.Connection, None]:
#     """
#     Async context manager wrapper for safe acquire/release:

#         async with connection_ctx() as conn:
#             await conn.fetch(...)
#     """
#     conn = None
#     try:
#         conn = await get_connection()
#         yield conn
#     finally:
#         if conn is not None:
#             await release_connection(conn)


# if __name__ == "__main__":
#     import asyncio

#     async def _test():
#         try:
#             await init_db_pool()
#             logger.info("Pool present: %s", _pool is not None)
#             ok = await health_check()
#             logger.info("Health check OK: %s", ok)

#             async with connection_ctx() as conn:
#                 row = await conn.fetchrow(
#                     "SELECT current_database() AS db, current_user AS user;"
#                 )
#                 logger.info("Connected to DB=%s user=%s", row["db"], row["user"])

#         except Exception as exc:
#             logger.exception("DB test failed: %s", exc)
#         finally:
#             await close_pool()

#     asyncio.run(_test())
