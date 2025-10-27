import asyncpg
import os
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional
from ..core.metrics import RequestMetrics

logger = logging.getLogger(__name__)

_DATABASE_URL = os.getenv("DATABASE_URL")

class AsyncPostgreSQLStorage:
    def __init__(self, database_url: Optional[str] = None, min_size: int = 1, max_size: int = 10):
        self._db_url = database_url or _DATABASE_URL
        if not self._db_url:
            raise RuntimeError("Set DATABASE_URL env var for async storage (or pass database_url to constructor)")
        self._pool = None
        self.min_size = min_size
        self.max_size = max_size

    async def init_pool(self):
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self._db_url, min_size=self.min_size, max_size=self.max_size)
            logger.info("AsyncPostgreSQLStorage pool created")

    async def store_metrics(self, metrics: RequestMetrics) -> bool:
        await self.init_pool()
        try:
            if getattr(metrics, "cache_hit", False) and not getattr(metrics, "error", False):
                return True
        except Exception:
            pass

        try:
            cost = getattr(metrics, "estimated_cost_usd", 0.0) or 0.0
            cost_dec = Decimal(str(cost)).quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
        except Exception:
            cost_dec = Decimal("0.00000000")

        params = None
        try:
            model_val = getattr(metrics, "model", None)
            if model_val is None or (isinstance(model_val, str) and model_val.strip() == ""):
                model_val = "unknown-model"

            ts_val = metrics.timestamp or datetime.now(timezone.utc)

            params = [
                metrics.request_id,
                ts_val,
                metrics.total_latency_ms,
                metrics.preprocessing_ms,
                metrics.llm_api_call_ms,
                metrics.postprocessing_ms,
                metrics.metrics_export_ms,
                metrics.input_tokens,
                metrics.output_tokens,
                metrics.total_tokens,
                cost_dec,
                model_val,
                metrics.endpoint,
                metrics.user_id,
                bool(metrics.error) if metrics.error is not None else False,
                metrics.error_message,
                metrics.error_type,
                metrics.status_code if metrics.status_code is not None else 200,
                metrics.request_size_bytes if metrics.request_size_bytes is not None else 0,
                metrics.response_size_bytes if metrics.response_size_bytes is not None else 0,
            ]

            async with self._pool.acquire() as conn:
                try:
                    ctx = await conn.fetchrow("SELECT current_database() AS db, current_user AS user")
                    logger.info("AsyncPostgreSQLStorage: inserting metrics -> DB=%s USER=%s", ctx["db"], ctx["user"])
                except Exception:
                    logger.exception("Could not fetch current DB/user for metrics writer")

                logger.debug("Metrics insert preview: table=llm_apm.llm_metrics params_sample=%s", str(params)[:800])

                # Execute the insert using explicit params unpacked
                await conn.execute(
                    """
                    INSERT INTO llm_apm.llm_metrics (
                      id, request_id, timestamp,
                      total_latency_ms, preprocessing_ms, llm_api_call_ms,
                      postprocessing_ms, metrics_export_ms,
                      input_tokens, output_tokens, total_tokens, estimated_cost_usd,
                      model, endpoint, user_id,
                      error, error_message, error_type,
                      status_code, request_size_bytes, response_size_bytes
                    ) VALUES (
                      gen_random_uuid(), $1, $2,
                      $3, $4, $5,
                      $6, $7,
                      $8, $9, $10, $11,
                      $12, $13, $14,
                      $15, $16, $17,
                      $18, $19, $20
                    ) ON CONFLICT (request_id) DO NOTHING
                    """,
                    *params
                )
            return True
        except Exception as e:
            try:
                logger.exception("Async store_metrics failed (params_sample=%s): %s", str(params)[:2000], e)
            except Exception:
                logger.exception("Async store_metrics failed (params unavailable): %s", e)
            return False

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None
