import logging

import uvicorn
from config import settings
from endpoints import register_storage_events
from endpoints import router as endpoints_router
from endpoints import storage as async_storage
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router.auth import router as auth_router

from llm_apm import LLMMonitor
from llm_apm.exporters.prometheus import PrometheusExporter, set_global_exporter
from llm_apm.middleware.fastapi import LLMAPMMiddleware, add_monitoring_endpoints
from llm_apm.storage.postgresql_async import AsyncPostgreSQLStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

app = FastAPI(title="LLM Application with APM", version="1.0.0")
app.include_router(auth_router, prefix="/auth")
app.include_router(endpoints_router)


@app.get("/healthz")
async def health():
    return {"status": "ok"}


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    storage = async_storage
    try:
        register_storage_events(app)
        logger.info("Registered async storage startup/shutdown handlers")
    except Exception as e:
        logger.warning(f"Failed to register async storage handlers: {e}")
except Exception as e:
    logger.warning(f"Async storage unavailable: {e}")
    storage = None

try:
    exporter = PrometheusExporter(start_http_server_flag=False)
    set_global_exporter(exporter)
    logger.info("Prometheus exporter initialized")
except Exception as e:
    logger.warning(f"Failed to initialize Prometheus exporter: {e}")
    exporter = None

monitor = LLMMonitor(storage=storage, exporter=exporter)

app.add_middleware(
    LLMAPMMiddleware,
    monitor=monitor,
    storage=storage,
    exporter=exporter,
    enable_storage=storage is not None,
    enable_prometheus=exporter is not None,
)


try:
    add_monitoring_endpoints(app)
    logger.info(
        "LLM-APM monitoring endpoints registered (metrics/health/llm-apm/status)"
    )
except Exception as e:
    logger.warning(f"Failed to add monitoring endpoints: {e}")


@app.on_event("startup")
async def startup_event():
    try:
        from llm_apm.utils.cache import init_redis

        await init_redis()
        logger.info("Cache Redis initialized")
    except Exception as e:
        logger.warning(f"init_redis failed: {e}")

    try:
        from llm_apm.utils.quota import init_quota_redis

        await init_quota_redis()
        logger.info("Quota Redis initialized")
    except Exception as e:
        logger.warning(f"init_quota_redis failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(settings.PORT),
        reload=bool(settings.RELOAD),
        log_level="info",
    )
