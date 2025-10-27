import logging
import os  
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from datetime import datetime, timezone
import uuid

from ..core.monitor import LLMMonitor, set_global_monitor, get_global_monitor
from ..storage.postgresql import PostgreSQLStorage, create_storage_from_config
from ..exporters.prometheus import PrometheusExporter, set_global_exporter, get_global_exporter
from ..config.settings import config

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = getattr(config, "model_name", None) or os.environ.get("MODEL_NAME")
if DEFAULT_MODEL_NAME:
    logger.info(f"LLM APM: using default MODEL_NAME='{DEFAULT_MODEL_NAME}' for metrics if handlers don't override it")
else:
    logger.debug("LLM APM: no default MODEL_NAME set; metrics without model will show 'unknown'")

class LLMAPMMiddleware(BaseHTTPMiddleware):
    
    def __init__(
        self,
        app,
        monitor: Optional[LLMMonitor] = None,
        storage: Optional[PostgreSQLStorage] = None,
        exporter: Optional[PrometheusExporter] = None,
        enable_storage: bool = True,
        enable_prometheus: bool = True,
        monitored_endpoints: Optional[list] = None
    ):
        super().__init__(app)
        if enable_storage and storage is None:
            try:
                storage = create_storage_from_config()
                logger.info("PostgreSQL storage initialized from config")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL storage: {e}")
                storage = None

        if enable_prometheus:
            try:
                if exporter is not None:
                    set_global_exporter(exporter)
                    logger.info("Using provided Prometheus exporter instance")
                else:
                    existing = get_global_exporter()
                    if existing is not None:
                        exporter = existing
                        logger.info("Using existing global Prometheus exporter")
                    else:
                        exporter = PrometheusExporter(start_http_server_flag=False)
                        set_global_exporter(exporter)
                        logger.info("Prometheus exporter initialized (served via FastAPI)")
            except Exception as e:
                logger.warning(f"Failed to initialize Prometheus exporter: {e}")
                exporter = None

        try:
            self.monitor = monitor or LLMMonitor(storage=storage, exporter=exporter)
            set_global_monitor(self.monitor)
        except Exception as e:
            logger.error(f"Failed to initialize LLMMonitor: {e}")
            self.monitor = None

        self.monitored_endpoints = monitored_endpoints or ["/generate"]
        logger.info(f"LLM APM Middleware initialized, monitoring endpoints: {self.monitored_endpoints}")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and add monitoring (initialise & clear decorator ContextVar)."""
        if not self._should_monitor_endpoint(request.url.path):
            logger.debug(f"Endpoint not monitored: {request.url.path}")
            return await call_next(request)

        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        user_id = request.headers.get("x-user-id")
        session_id = request.headers.get("x-session-id")

        request_monitor = None
        try:
            if self.monitor:
                request_monitor = self.monitor.start_request_monitoring(
                    endpoint=request.url.path,
                    model=DEFAULT_MODEL_NAME,  
                    prompt="",
                    user_id=None,     
                    session_id=session_id,
                    request_id=request_id
                )
            else:
                logger.debug("No monitor available; skipping request monitoring")
        except Exception as e:
            logger.error(f"Failed to start request monitoring: {e}")
            request_monitor = None

        if request_monitor is not None:
            request.state.llm_monitor = request_monitor
            request.state.request_id = request_id

            try:
                if getattr(request_monitor, "metrics", None):
                    header_user = (
                        request.headers.get("x-user-id")
                        or request.headers.get("x-userid")
                        or request.headers.get("x-forwarded-user")
                    )
                    state_user = None
                    state_user_name = None
                    try:
                        cur = getattr(request.state, "current_user", None)
                        if isinstance(cur, dict):
                            state_user = cur.get("id") or cur.get("user_id") or cur.get("uid")
                            state_user_name = cur.get("username") or cur.get("name") or cur.get("email")
                        elif cur:
                            state_user = getattr(cur, "id", None) or getattr(cur, "uuid", None) or str(cur)
                            state_user_name = getattr(cur, "username", None) or getattr(cur, "name", None)
                    except Exception:
                        state_user = None
                        state_user_name = None

                    session_user = request.cookies.get("session_id") or request.headers.get("x-session-id")

                    chosen_user = state_user or header_user or session_user or None

                    chosen_name = state_user_name or request.headers.get("x-user-name") or request.headers.get("x-username") or None

                    try:
                        if chosen_user:
                            request_monitor.metrics.user_id = str(chosen_user)
                    except Exception:
                        pass

                    try:
                        if chosen_name:
                            request_monitor.metrics.user_name = str(chosen_name)
                    except Exception:
                        pass
            except Exception:
                pass

        exporter = get_global_exporter()
        try:
            if exporter and getattr(request_monitor, "sample", False) and getattr(request_monitor, "metrics", None):
                metric_model = getattr(request_monitor.metrics, "model", None)
                model_label = metric_model or DEFAULT_MODEL_NAME or "unknown"
                exporter.increment_active_requests(model=model_label, endpoint=request.url.path)
        except Exception as e:
            logger.debug(f"Could not increment active_requests metric: {e}")

        try:
            from ..core.decorators import current_step_timer, clear_step_context
        except Exception:
            current_step_timer = None
            clear_step_context = None

        if request_monitor is not None and current_step_timer is not None:
            try:
                current_step_timer.set({})
            except Exception as e:
                logger.debug(f"Could not set current_step_timer: {e}")

        def _populate_model_from_request():
            try:
                if not request_monitor or not getattr(request_monitor, "metrics", None):
                    return
                model_name = getattr(request.state, "current_model", None)
                if model_name:
                    request_monitor.metrics.model = str(model_name)
                    return
                header_model = (
                    request.headers.get("x-model")
                    or request.headers.get("x-llm-model")
                    or request.headers.get("x-openai-model")
                )
                if header_model:
                    request_monitor.metrics.model = str(header_model)
                    return
                existing = getattr(request_monitor.metrics, "model", None)
                if existing and existing not in ("", "unknown", None):
                    return
            except Exception:
                pass

        try:
            if request_monitor is not None:
                logger.debug(f"Entering request monitor context: request_id={request_id}, sampled={getattr(request_monitor, 'sample', False)}")
                with request_monitor:
                    response = await call_next(request)
            else:
                response = await call_next(request)

            try:
                _populate_model_from_request()
            except Exception:
                logger.debug("Failed to populate model into request_monitor.metrics", exc_info=True)

            try:
                if exporter and getattr(request_monitor, "sample", False) and getattr(request_monitor, "metrics", None):
                    metric_model = getattr(request_monitor.metrics, "model", None)
                    model_label = metric_model or DEFAULT_MODEL_NAME or "unknown"
                    exporter.decrement_active_requests(model=model_label, endpoint=request.url.path)
            except Exception as e:
                logger.debug(f"Could not decrement active_requests metric: {e}")

            return response

        except Exception as e:
            try:
                _populate_model_from_request()
            except Exception:
                pass

            try:
                if request_monitor is not None:
                    request_monitor.record_error(e)
            except Exception as rec_err:
                logger.debug(f"Failed to record error in request_monitor: {rec_err}")

            try:
                if exporter and getattr(request_monitor, "sample", False) and getattr(request_monitor, "metrics", None):
                    metric_model = getattr(request_monitor.metrics, "model", None)
                    model_label = metric_model or DEFAULT_MODEL_NAME or "unknown"
                    exporter.decrement_active_requests(model=model_label, endpoint=request.url.path)
            except Exception:
                pass

            raise

        finally:
            if request_monitor is not None and 'clear_step_context' in locals() and clear_step_context is not None:
                try:
                    clear_step_context()
                except Exception:
                    pass


    def _should_monitor_endpoint(self, path: str) -> bool:
        """Check if endpoint should be monitored."""
        if not self.monitored_endpoints:
            return True
        return any(endpoint in path for endpoint in self.monitored_endpoints)


def add_metrics_endpoint(app, path: str = None):
    """Add Prometheus metrics endpoint to FastAPI app"""
    endpoint_path = path or getattr(config, "metrics_endpoint", "/metrics")

    @app.get(endpoint_path)
    async def metrics():
        """Prometheus metrics endpoint"""
        try:
            exporter = get_global_exporter()
            if exporter:
                metrics_data = exporter.get_metrics()
                return Response(
                    content=metrics_data,
                    media_type="text/plain; version=0.0.4; charset=utf-8"
                )
            else:
                return JSONResponse(
                    content={"error": "Prometheus exporter not available"},
                    status_code=503
                )
        except Exception as e:
            logger.error(f"Failed to serve metrics: {e}")
            return JSONResponse(
                content={"error": "Failed to retrieve metrics", "detail": str(e)},
                status_code=500
            )


def add_health_endpoint(app, path: str = "/health"):
    """Add health check endpoint to FastAPI app"""
    @app.get(path)
    async def health():
        """Health check endpoint"""
        try:
            monitor = get_global_monitor()
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "components": {
                    "monitor": "healthy" if monitor else "unavailable",
                    "storage": "unknown",
                    "exporter": "unknown"
                }
            }
            if monitor and monitor.storage:
                try:
                    healthy = monitor.storage.health_check()
                    health_status["components"]["storage"] = "healthy" if healthy else "unhealthy"
                    if not healthy:
                        health_status["status"] = "degraded"
                except Exception:
                    health_status["components"]["storage"] = "error"
                    health_status["status"] = "degraded"

            exporter = get_global_exporter()
            if exporter:
                health_status["components"]["exporter"] = "healthy"
            else:
                health_status["components"]["exporter"] = "unavailable"

            status_code = 200 if health_status["status"] == "healthy" else 503
            return JSONResponse(content=health_status, status_code=status_code)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return JSONResponse(
                content={
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                status_code=500
            )


def add_monitoring_endpoints(app):
    add_metrics_endpoint(app, path=getattr(config, "metrics_endpoint", "/metrics"))
    add_health_endpoint(app, path="/health")

    @app.get("/llm-apm/status")
    async def monitoring_status():
        try:
            monitor = get_global_monitor()
            if not monitor:
                return JSONResponse(content={"error": "Monitor not available"}, status_code=503)
            summary = monitor.get_metrics_summary(time_window_minutes=60)
            model_stats = monitor.get_model_stats()
            return JSONResponse(content={
                "status": "active",
                "config": {
                    "sampling_rate": getattr(config, "sampling_rate", None),
                    "metrics_endpoint": getattr(config, "metrics_endpoint", "/metrics")
                },
                "recent_metrics": summary,
                "model_statistics": model_stats,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to get monitoring status: {e}")
            return JSONResponse(content={"error": str(e)}, status_code=500)
