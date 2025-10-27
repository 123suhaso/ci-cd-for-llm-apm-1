
import logging
import time
import threading
from typing import Optional
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry,
    generate_latest, REGISTRY, start_http_server
)
from ..core.metrics import RequestMetrics
from ..config.settings import config
import hashlib
import os

logger = logging.getLogger(__name__)


def _user_hash_from_id(user_id: Optional[str]) -> str:
    try:
        if not user_id:
            return "unknown"
        h = hashlib.sha256(str(user_id).encode("utf-8")).hexdigest()
        return h[:8]
    except Exception:
        return "unknown"


def _coerce_model_label(raw_model: Optional[str]) -> str:
    try:
        if not raw_model:
            return "unknown"
        s = str(raw_model).strip()
        if not s or s.lower() in ("none", "null"):
            return "unknown"
        return s
    except Exception:
        return "unknown"


class PrometheusExporter:
    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
        port: Optional[int] = None,
        host: Optional[str] = None,
        start_http_server_flag: bool = True,
    ):
        if registry is not None:
            self.registry = registry
        else:
            self.registry = REGISTRY if start_http_server_flag else CollectorRegistry()
        self.port = port or getattr(config, "prometheus_port", 8000)
        self.host = host or getattr(config, "prometheus_host", "0.0.0.0")
        self.start_http_server_flag = start_http_server_flag
        self.server_thread = None
        self._create_metrics()

    def _create_metrics(self):
        self.request_total = Counter(
            "llm_requests_total",
            "Total number of LLM requests",
            ["model", "endpoint", "status", "experiment"],
            registry=self.registry,
        )
        self.request_duration = Histogram(
            "llm_request_duration_seconds",
            "LLM request duration in seconds",
            ["model", "endpoint", "experiment"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
            registry=self.registry,
        )
        self.preprocessing_duration = Histogram(
            "llm_preprocessing_duration_seconds",
            "LLM preprocessing duration in seconds",
            ["model", "endpoint", "experiment"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")),
            registry=self.registry,
        )
        self.llm_api_call_duration = Histogram(
            "llm_api_call_duration_seconds",
            "LLM API call duration in seconds",
            ["model", "endpoint", "experiment"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, float("inf")),
            registry=self.registry,
        )
        self.postprocessing_duration = Histogram(
            "llm_postprocessing_duration_seconds",
            "LLM postprocessing duration in seconds",
            ["model", "endpoint", "experiment"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")),
            registry=self.registry,
        )
        self.metrics_export_duration = Histogram(
            "llm_metrics_export_duration_seconds",
            "LLM metrics export duration in seconds",
            ["model", "endpoint", "experiment"],
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, float("inf")),
            registry=self.registry,
        )

        self.tokens_total = Counter(
            "llm_tokens_total",
            "Total number of tokens processed",
            ["model", "endpoint", "type", "experiment"],
            registry=self.registry,
        )
        self.tokens_per_request = Histogram(
            "llm_tokens_per_request",
            "Number of tokens per request",
            ["model", "endpoint", "type", "experiment"],
            buckets=(10, 50, 100, 500, 1000, 2000, 4000, 8000, float("inf")),
            registry=self.registry,
        )
        self.cost_total = Counter(
            "llm_cost_total_usd",
            "Total cost in USD",
            ["model", "endpoint", "experiment"],
            registry=self.registry,
        )
        self.cost_per_request = Histogram(
            "llm_cost_per_request_usd",
            "Cost per request in USD",
            ["model", "endpoint", "experiment"],
            buckets=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, float("inf")),
            registry=self.registry,
        )
        self.cache_hits = Counter(
            "llm_cache_hits_total",
            "Total number of cache hits",
            ["endpoint", "experiment"],
            registry=self.registry,
        )
        self.cache_misses = Counter(
            "llm_cache_misses_total",
            "Total number of cache misses",
            ["endpoint", "experiment"],
            registry=self.registry,
        )
        self.sampling_rate = Gauge(
            "llm_sampling_rate",
            "Configured sampling rate",
            ["endpoint"],
            registry=self.registry,
        )
        self.active_requests = Gauge(
            "llm_active_requests",
            "Number of currently active LLM requests",
            ["model", "endpoint", "experiment"],
            registry=self.registry,
        )
        self.model_usage = Counter(
            "llm_model_usage_total",
            "Total usage per model",
            ["model", "experiment"],
            registry=self.registry,
        )
        self.error_rate = Gauge(
            "llm_error_rate",
            "Current error rate (0-1)",
            ["model", "endpoint", "experiment"],
            registry=self.registry,
        )
        self.response_size_bytes = Histogram(
            "llm_response_size_bytes",
            "Response size in bytes",
            ["model", "endpoint", "experiment"],
            buckets=(100, 500, 1000, 5000, 10000, 50000, 100000, float("inf")),
            registry=self.registry,
        )

        self.apm_request_total = Counter(
            "llm_apm_request_total",
            "LLM APM: total requests",
            ["model", "endpoint", "status", "user_hash", "experiment"],
            registry=self.registry,
        )
        self.apm_request_duration = Histogram(
            "llm_apm_request_duration_seconds",
            "LLM APM: request duration (seconds)",
            ["model", "endpoint", "experiment"],
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, float("inf")),
            registry=self.registry,
        )
        self.apm_step_duration = Histogram(
            "llm_apm_step_duration_seconds",
            "LLM APM: step duration seconds (step=preprocess|api_call|postprocess|metrics_export)",
            ["step", "model", "endpoint", "experiment"],
            buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, float("inf")),
            registry=self.registry,
        )
        self.apm_tokens_prompt_total = Counter(
            "llm_apm_tokens_prompt_total",
            "LLM APM: total prompt tokens",
            ["model", "endpoint", "type", "user_hash", "experiment"],
            registry=self.registry,
        )
        self.apm_tokens_completion_total = Counter(
            "llm_apm_tokens_completion_total",
            "LLM APM: total completion tokens",
            ["model", "endpoint", "type", "user_hash", "experiment"],
            registry=self.registry,
        )
        self.apm_tokens_total = Counter(
            "llm_apm_tokens_total",
            "LLM APM: total tokens (prompt+completion)",
            ["model", "endpoint", "type", "user_hash", "experiment"],
            registry=self.registry,
        )
        self.apm_errors_total = Counter(
            "llm_apm_errors_total",
            "LLM APM: total errors",
            ["model", "endpoint", "error_type", "user_hash", "experiment"],
            registry=self.registry,
        )

        self.apm_cost_usd_total = Counter(
            "llm_apm_cost_usd_total",
            "LLM APM: total cost in USD",
            ["model", "endpoint", "type", "user_hash", "experiment"],
            registry=self.registry,
        )
        self.apm_cost_per_request_gauge = Gauge(
            "llm_apm_cost_per_request_usd",
            "LLM APM: cost per request in USD (last observed)",
            ["model", "endpoint", "type", "user_hash", "experiment"],
            registry=self.registry,
        )

        self.apm_slo_ok = Gauge(
            "llm_apm_slo_ok",
            "LLM APM: last request met SLO (1=ok,0=not ok)",
            ["model", "endpoint", "experiment"],
            registry=self.registry,
        )
        self.llm_apm_sampling_rate = Gauge(
            "llm_apm_sampling_rate",
            "LLM APM: sampling rate",
            ["endpoint"],
            registry=self.registry,
        )
        self.llm_apm_cache_hit_total = Counter(
            "llm_apm_cache_hit_total",
            "LLM APM: cache hits",
            ["endpoint", "experiment"],
            registry=self.registry,
        )
        self.llm_apm_cache_miss_total = Counter(
            "llm_apm_cache_miss_total",
            "LLM APM: cache misses",
            ["endpoint", "experiment"],
            registry=self.registry,
        )

        self.apm_response_quality = Gauge(
            "llm_apm_response_quality",
            "LLM APM: response quality score (0..1 or custom scale)",
            ["model", "endpoint", "user_hash", "experiment"],
            registry=self.registry,
        )

        logger.info(
            "Prometheus metrics created successfully (registry=%s)",
            "global" if self.registry is REGISTRY else "private",
        )

    def start(self):
        if not self.start_http_server_flag:
            logger.info("PrometheusExporter not starting standalone HTTP server")
            return
        try:
            if self.server_thread is None or not self.server_thread.is_alive():
                self.server_thread = threading.Thread(
                    target=start_http_server, args=(self.port, self.host), daemon=True
                )
                self.server_thread.start()
                logger.info(f"Prometheus server started on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start Prometheus server: {e}")

    def _safe_user_label(self, metrics_obj: Optional[RequestMetrics]) -> str:
        try:
            if not metrics_obj:
                return "unknown"
            existing_hash = getattr(metrics_obj, "user_hash", None)
            if existing_hash:
                return str(existing_hash)
            stable_user_id = getattr(metrics_obj, "user_id", None)
            if stable_user_id:
                return _user_hash_from_id(stable_user_id)
            return "unknown"
        except Exception:
            return "unknown"

    def _maybe_attach_exemplar(self, histogram, value_seconds: float, labels: dict):
        try:
            if os.getenv("ENABLE_OTEL", "false").lower() in ("1", "true", "yes"):
                pass
        except Exception:
            pass

    def record_request(self, metrics: RequestMetrics) -> float:
        t0 = time.perf_counter()
        try:
            user_label = self._safe_user_label(metrics)
            try:
                if not getattr(metrics, "user_hash", None) and user_label and user_label != "unknown":
                    metrics.user_hash = user_label
            except Exception:
                pass

            experiment = getattr(metrics, "experiment", "control") or "control"
            model_label = _coerce_model_label(getattr(metrics, "model", None))

            labels_common = {
                "model": model_label,
                "endpoint": metrics.endpoint or "unknown",
                "experiment": experiment,
            }

            status = "error" if metrics.error else "success"
            try:
                self.request_total.labels(**labels_common, status=status).inc()
            except Exception:
                pass

            try:
                self.request_duration.labels(**labels_common).observe(metrics.total_latency_ms / 1000.0)
                self.preprocessing_duration.labels(**labels_common).observe(metrics.preprocessing_ms / 1000.0)
                self.llm_api_call_duration.labels(**labels_common).observe(metrics.llm_api_call_ms / 1000.0)
                self.postprocessing_duration.labels(**labels_common).observe(metrics.postprocessing_ms / 1000.0)
                self.metrics_export_duration.labels(**labels_common).observe(metrics.metrics_export_ms / 1000.0)
            except Exception:
                logger.debug("Legacy duration observation failed", exc_info=True)

            try:
                self.tokens_total.labels(**labels_common, type="input").inc(metrics.input_tokens or 0)
                self.tokens_total.labels(**labels_common, type="output").inc(metrics.output_tokens or 0)
                self.tokens_per_request.labels(**labels_common, type="input").observe(metrics.input_tokens or 0)
                self.tokens_per_request.labels(**labels_common, type="output").observe(metrics.output_tokens or 0)
            except Exception:
                logger.debug("Legacy token metrics failed", exc_info=True)

            try:
                self.cost_total.labels(**labels_common).inc(metrics.estimated_cost_usd or 0.0)
                self.cost_per_request.labels(**labels_common).observe(metrics.estimated_cost_usd or 0.0)
            except Exception:
                logger.debug("Legacy cost metrics failed", exc_info=True)

            try:
                if (metrics.response_size_bytes or 0) > 0:
                    self.response_size_bytes.labels(**labels_common).observe(metrics.response_size_bytes)
            except Exception:
                pass

            try:
                self.apm_request_total.labels(
                    model=model_label,
                    endpoint=metrics.endpoint or "unknown",
                    status=status,
                    user_hash=user_label,
                    experiment=experiment
                ).inc()
            except Exception:
                logger.debug("apm_request_total inc failed", exc_info=True)

            try:
                self.apm_request_duration.labels(**labels_common).observe(metrics.total_latency_ms / 1000.0)
                self._maybe_attach_exemplar(self.apm_request_duration, metrics.total_latency_ms / 1000.0, labels_common)
            except Exception:
                logger.debug("apm_request_duration observe failed", exc_info=True)

            step_map = {
                "preprocessing": metrics.preprocessing_ms,
                "llm_api_call": metrics.llm_api_call_ms,
                "postprocessing": metrics.postprocessing_ms,
                "metrics_export": metrics.metrics_export_ms,
            }
            for step_name, ms in step_map.items():
                try:
                    self.apm_step_duration.labels(step=step_name, model=model_label, endpoint=metrics.endpoint or "unknown", experiment=experiment).observe((ms or 0.0) / 1000.0)
                except Exception:
                    logger.debug("apm_step_duration observe failed for %s", step_name, exc_info=True)

            try:
                token_type = "request"
                if getattr(metrics, "cache_hit", False) or getattr(metrics, "from_cache", False):
                    token_type = "cache"

                self.apm_tokens_prompt_total.labels(
                    model=model_label, endpoint=metrics.endpoint or "unknown", type=token_type, user_hash=user_label, experiment=experiment
                ).inc(metrics.input_tokens or 0)
                self.apm_tokens_completion_total.labels(
                    model=model_label, endpoint=metrics.endpoint or "unknown", type=token_type, user_hash=user_label, experiment=experiment
                ).inc(metrics.output_tokens or 0)
                self.apm_tokens_total.labels(
                    model=model_label, endpoint=metrics.endpoint or "unknown", type=token_type, user_hash=user_label, experiment=experiment
                ).inc(metrics.total_tokens or ((metrics.input_tokens or 0) + (metrics.output_tokens or 0)))
            except Exception:
                logger.debug("apm token counters failed", exc_info=True)

            if metrics.error:
                err_type = (metrics.error_type or "unknown").lower()
                try:
                    self.apm_errors_total.labels(
                        model=model_label,
                        endpoint=metrics.endpoint or "unknown",
                        error_type=err_type,
                        user_hash=user_label,
                        experiment=experiment
                    ).inc()
                except Exception:
                    logger.debug("apm_errors_total inc failed", exc_info=True)

            try:
                self.apm_cost_usd_total.labels(
                    model=model_label,
                    endpoint=metrics.endpoint or "unknown",
                    type=token_type,
                    user_hash=user_label,
                    experiment=experiment
                ).inc(metrics.estimated_cost_usd or 0.0)
                self.apm_cost_per_request_gauge.labels(
                    model=model_label,
                    endpoint=metrics.endpoint or "unknown",
                    type=token_type,
                    user_hash=user_label,
                    experiment=experiment
                ).set(metrics.estimated_cost_usd or 0.0)
            except Exception:
                logger.debug("apm cost metrics failed", exc_info=True)

            try:
                slo_ok = 1 if metrics.total_latency_ms <= getattr(config, "latency_threshold_ms", 3000) else 0
                self.apm_slo_ok.labels(model=model_label, endpoint=metrics.endpoint or "unknown", experiment=experiment).set(slo_ok)
            except Exception:
                pass

            try:
                self.sampling_rate.labels(endpoint=metrics.endpoint or "unknown").set(getattr(config, "sampling_rate", 1.0))
                self.llm_apm_sampling_rate.labels(endpoint=metrics.endpoint or "unknown").set(getattr(config, "sampling_rate", 1.0))
            except Exception:
                pass

            try:
                rq = getattr(metrics, "response_quality", None)
                if rq is not None:
                    self.apm_response_quality.labels(
                        model=model_label, endpoint=metrics.endpoint or "unknown", user_hash=user_label, experiment=experiment
                    ).set(float(rq))
            except Exception:
                logger.debug("apm_response_quality set failed", exc_info=True)

        except Exception as e:
            logger.error(f"Failed to record Prometheus metrics: {e}", exc_info=True)
        finally:
            return time.perf_counter() - t0

    def get_metrics(self) -> str:
        try:
            return generate_latest(self.registry).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to generate Prometheus metrics: {e}", exc_info=True)
            return ""

    def increment_active_requests(self, model: str, endpoint: str, experiment: str = "control"):
        try:
            safe_model = _coerce_model_label(model)
            self.active_requests.labels(model=safe_model, endpoint=endpoint, experiment=experiment).inc()
        except Exception as e:
            logger.error(f"Failed to increment active requests: {e}")

    def decrement_active_requests(self, model: str, endpoint: str, experiment: str = "control"):
        try:
            safe_model = _coerce_model_label(model)
            self.active_requests.labels(model=safe_model, endpoint=endpoint, experiment=experiment).dec()
        except Exception as e:
            logger.error(f"Failed to decrement active requests: {e}")

    def update_error_rate(self, model: str, endpoint: str, experiment: str, error_rate: float):
        try:
            safe_model = _coerce_model_label(model)
            self.error_rate.labels(model=safe_model, endpoint=endpoint, experiment=experiment).set(error_rate)
        except Exception as e:
            logger.error(f"Failed to update error rate: {e}")

    def shutdown(self):
        logger.info("Prometheus exporter shutdown")


_GLOBAL_EXPORTER = None


def set_global_exporter(exporter):
    global _GLOBAL_EXPORTER
    _GLOBAL_EXPORTER = exporter


def get_global_exporter():
    return _GLOBAL_EXPORTER
