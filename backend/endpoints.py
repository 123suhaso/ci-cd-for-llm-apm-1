# endpoints.py
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from decimal import ROUND_DOWN, Decimal
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

try:
    from openai import AsyncAzureOpenAI
except Exception:
    AsyncAzureOpenAI = None

try:
    from ollama import AsyncClient as OllamaAsyncClient  # optional

    _HAS_OLLAMA = True
except Exception:
    OllamaAsyncClient = None
    _HAS_OLLAMA = False

try:
    import httpx
except Exception:
    httpx = None

from config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_ENDPOINT,
    settings,
)
from router.auth import get_current_user

from llm_apm.core.decorators import get_current_step_metrics, step
from llm_apm.exporters.prometheus import PrometheusExporter, get_global_exporter
from llm_apm.storage.postgresql_async import AsyncPostgreSQLStorage
from llm_apm.utils.cache import get_key as cache_get_key
from llm_apm.utils.cache import make_key as cache_make_key
from llm_apm.utils.cache import set_key as cache_set_key
from llm_apm.utils.cost_calculator import cost_calculator
from llm_apm.utils.prompt_utils import build_prompt_and_truncate
from llm_apm.utils.quota import (
    increment_and_check,
    init_quota_redis,
    is_in_cooldown,
    set_cooldown,
)
from llm_apm.utils.sampler import decide_variant
from llm_apm.utils.token_counter import token_counter

logger = logging.getLogger("endpoints")
if not logger.handlers:
    logging.basicConfig(level=logging.DEBUG)
router = APIRouter()

DATABASE_URL = os.getenv("DATABASE_URL")
storage = AsyncPostgreSQLStorage(database_url=DATABASE_URL)


def _canonicalize_prompt(p: Optional[str]) -> str:
    if not isinstance(p, str):
        return ""
    return " ".join(p.split()).strip()


def _normalize_provider_model(provider: Optional[str], model: Optional[str]):
    try:
        p = provider.lower().strip() if provider and isinstance(provider, str) else None
    except Exception:
        p = None
    try:
        m = model.strip() if model and isinstance(model, str) else None
    except Exception:
        m = None
    return p, m


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0
    session_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None


class GenerateResponse(BaseModel):
    response: str
    metrics: Dict[str, Any]


MAX_PROMPT_TOKENS = settings.MAX_PROMPT_TOKENS
CACHE_TTL = settings.CACHE_TTL_SECONDS
MAX_TOKENS_LIMIT = settings.MAX_TOKENS_LIMIT
DEFAULT_PROMPT_MAX_CHARS = settings.PROMPT_MAX_CHARS
MAX_ATTEMPTS = settings.LLM_MAX_ATTEMPTS
PER_ATTEMPT_TIMEOUT = settings.LLM_PER_ATTEMPT_TIMEOUT
BASE_BACKOFF = settings.LLM_BASE_BACKOFF

DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "ollama").lower().strip() or "ollama"

if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT and AsyncAzureOpenAI is not None:
    azure_client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
else:
    azure_client = None
    if not (AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT):
        logger.info(
            "Azure OpenAI env vars missing; azure models won't work unless configured."
        )

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
ollama_client = None
if _HAS_OLLAMA:
    try:
        ollama_client = OllamaAsyncClient()
    except Exception as e:
        ollama_client = None
        logger.debug("Failed to init OllamaAsyncClient: %s", e)


def _select_provider_for_model(model_name: Optional[str]) -> str:
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        return DEFAULT_PROVIDER
    mn = model_name.lower().strip()
    if mn.startswith("llama") or "llama" in mn or "ollama" in mn or ":" in mn:
        return "ollama"
    if "gpt-" in mn or mn.startswith("gpt") or "gpt" in mn:
        return "azure"
    return DEFAULT_PROVIDER


def _find_any_string_in_obj(obj, max_len=10000):
    try:
        if obj is None:
            return None
        if isinstance(obj, str):
            s = obj.strip()
            return s if s else None
        if isinstance(obj, (int, float, bool)):
            return str(obj)
        if isinstance(obj, dict):
            for k in (
                "response",
                "output",
                "text",
                "generated_text",
                "content",
                "message",
                "result",
                "answer",
                "generation",
            ):
                if k in obj:
                    val = _find_any_string_in_obj(obj[k], max_len=max_len)
                    if val:
                        return val
            for _, v in obj.items():
                val = _find_any_string_in_obj(v, max_len=max_len)
                if val:
                    return val
            return None
        if isinstance(obj, (list, tuple)):
            for item in obj:
                val = _find_any_string_in_obj(item, max_len=max_len)
                if val:
                    return val
            return None
        for attr in (
            "choices",
            "outputs",
            "data",
            "result",
            "text",
            "content",
            "output_text",
        ):
            try:
                if hasattr(obj, attr):
                    val = getattr(obj, attr)
                    found = _find_any_string_in_obj(val, max_len=max_len)
                    if found:
                        return found
            except Exception:
                continue
    except Exception:
        return None
    return None


def _extract_response_and_usage(
    sdk_response, chosen_model: str, fallback_input_tokens: int
):
    content = ""
    try:
        if isinstance(sdk_response, dict):
            content = (
                sdk_response.get("response")
                or sdk_response.get("output")
                or sdk_response.get("text")
                or ""
            )
            if not content:
                content = (
                    sdk_response.get("generated_text")
                    or sdk_response.get("generated")
                    or ""
                )
            if (
                not content
                and "choices" in sdk_response
                and isinstance(sdk_response["choices"], (list, tuple))
                and sdk_response["choices"]
            ):
                ch0 = sdk_response["choices"][0]
                if isinstance(ch0, dict):
                    content = (
                        ch0.get("text")
                        or (ch0.get("message") and ch0["message"].get("content"))
                        or content
                    )
            if (
                not content
                and "outputs" in sdk_response
                and isinstance(sdk_response["outputs"], (list, tuple))
                and sdk_response["outputs"]
            ):
                out0 = sdk_response["outputs"][0]
                if isinstance(out0, dict):
                    cnt = (
                        out0.get("content")
                        or out0.get("text")
                        or out0.get("message")
                        or out0.get("output")
                    )
                    if cnt:
                        content = _find_any_string_in_obj(cnt) or content
    except Exception:
        pass

    if not (content and str(content).strip()):
        try:
            choices = getattr(sdk_response, "choices", None)
            if choices and isinstance(choices, (list, tuple)):
                ch0 = choices[0]
                try:
                    msg = getattr(ch0, "message", None)
                    if msg:
                        content = getattr(msg, "content", "") or content
                except Exception:
                    pass
                if not content:
                    try:
                        content = getattr(ch0, "text", "") or content
                    except Exception:
                        pass
            if not content:
                content = (
                    getattr(sdk_response, "output_text", "")
                    or getattr(sdk_response, "text", "")
                    or content
                )
        except Exception:
            pass

    if not (content and str(content).strip()):
        try:
            fallback = _find_any_string_in_obj(sdk_response)
            if fallback and isinstance(fallback, str):
                content = fallback
        except Exception:
            pass

    usage = {}
    try:
        if isinstance(sdk_response, dict):
            usage_candidate = (
                sdk_response.get("usage")
                or sdk_response.get("stats")
                or sdk_response.get("metadata")
                or {}
            )
        else:
            usage_candidate = getattr(sdk_response, "usage", None)
        if usage_candidate:
            usage = usage_candidate
    except Exception:
        usage = {}

    def _get_usage_field(u, key):
        try:
            if isinstance(u, dict):
                return u.get(key)
            return getattr(u, key)
        except Exception:
            return None

    prompt_tokens = _get_usage_field(usage, "prompt_tokens")
    completion_tokens = _get_usage_field(usage, "completion_tokens")
    total_tokens = _get_usage_field(usage, "total_tokens")

    if prompt_tokens is None:
        for alt in ("input_tokens", "promptToken", "promptTokens", "tokens_prompt"):
            v = _get_usage_field(usage, alt)
            if v is not None:
                prompt_tokens = v
                break
    if completion_tokens is None:
        for alt in (
            "output_tokens",
            "completionToken",
            "completionTokens",
            "tokens_completion",
        ):
            v = _get_usage_field(usage, alt)
            if v is not None:
                completion_tokens = v
                break
    if total_tokens is None:
        for alt in ("total_tokens", "totalToken", "totalTokens", "tokens_total"):
            v = _get_usage_field(usage, alt)
            if v is not None:
                total_tokens = v
                break

    try:
        prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
    except Exception:
        prompt_tokens = None
    try:
        completion_tokens = (
            int(completion_tokens) if completion_tokens is not None else None
        )
    except Exception:
        completion_tokens = None
    try:
        total_tokens = int(total_tokens) if total_tokens is not None else None
    except Exception:
        total_tokens = None

    if prompt_tokens is None:
        try:
            prompt_tokens = int(fallback_input_tokens or 0)
        except Exception:
            prompt_tokens = 0

    if completion_tokens is None:
        try:
            completion_tokens = int(
                token_counter.count_tokens(content or "", chosen_model)
            )
        except Exception:
            try:
                completion_tokens = max(0, (len(content or "") // 4))
            except Exception:
                completion_tokens = 0

    if total_tokens is None:
        try:
            total_tokens = int(prompt_tokens) + int(completion_tokens)
        except Exception:
            total_tokens = int((prompt_tokens or 0) + (completion_tokens or 0))

    try:
        prompt_tokens = int(prompt_tokens or 0)
    except Exception:
        prompt_tokens = 0
    try:
        completion_tokens = int(completion_tokens or 0)
    except Exception:
        completion_tokens = 0
    try:
        total_tokens = int(total_tokens or (prompt_tokens + completion_tokens))
    except Exception:
        total_tokens = prompt_tokens + completion_tokens

    return (content or "", prompt_tokens, completion_tokens, total_tokens)


async def _list_ollama_models() -> List[str]:
    models: List[str] = []
    if ollama_client is not None:
        try:
            if hasattr(ollama_client, "list") and callable(ollama_client.list):
                res = ollama_client.list()
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, (list, tuple)):
                    for entry in res:
                        if isinstance(entry, dict):
                            name = (
                                entry.get("name")
                                or entry.get("model")
                                or entry.get("id")
                            )
                            if name:
                                models.append(str(name))
            elif hasattr(ollama_client, "models") and callable(ollama_client.models):
                res = ollama_client.models()
                if asyncio.iscoroutine(res):
                    res = await res
                if isinstance(res, (list, tuple)):
                    for entry in res:
                        if isinstance(entry, dict):
                            name = entry.get("name") or entry.get("model")
                            if name:
                                models.append(name)
            if models:
                return sorted(list(set(models)))
        except Exception as e:
            logger.debug("Ollama async client model list attempt failed: %s", e)

    if httpx is None:
        logger.debug("httpx not installed; cannot query Ollama HTTP API for models")
        return []

    urls_to_try = [
        f"{OLLAMA_API_URL.rstrip('/')}/api/tags",
        f"{OLLAMA_API_URL.rstrip('/')}/api/models",
        f"{OLLAMA_API_URL.rstrip('/')}/api/list",
    ]
    async with httpx.AsyncClient(timeout=10.0) as httpc:
        for url in urls_to_try:
            try:
                resp = await httpc.get(url)
                if resp.status_code >= 400:
                    continue
                j = resp.json()
                if isinstance(j, dict):
                    if "models" in j and isinstance(j["models"], list):
                        for m in j["models"]:
                            if isinstance(m, dict):
                                nm = m.get("name") or m.get("model") or m.get("id")
                                if nm:
                                    models.append(str(nm))
                    elif "data" in j and isinstance(j["data"], list):
                        for item in j["data"]:
                            if isinstance(item, dict):
                                nm = item.get("name") or item.get("model")
                                if nm:
                                    models.append(str(nm))
                if isinstance(j, list):
                    for item in j:
                        if isinstance(item, dict):
                            nm = item.get("name") or item.get("model")
                            if nm:
                                models.append(str(nm))
                if models:
                    return sorted(list(set(models)))
            except Exception:
                continue
    return sorted(list(set(models)))


@router.get("/available_models")
async def available_models():
    provs: Dict[str, List[str]] = {"ollama": [], "azure": []}
    try:
        ollama_models = await _list_ollama_models()
        provs["ollama"] = ollama_models
    except Exception as e:
        logger.exception("Failed to list ollama models: %s", e)
        provs["ollama"] = []
    try:
        az_models = []
        if AZURE_OPENAI_DEPLOYMENT:
            az_models.append(AZURE_OPENAI_DEPLOYMENT)
        provs["azure"] = az_models
    except Exception:
        provs["azure"] = []
    return {"providers": provs, "default_provider": DEFAULT_PROVIDER}


@step("preprocessing")
async def preprocess_request(
    request: GenerateRequest,
    fastapi_request: Request,
    current_user: dict,
    request_monitor,
    do_quota_check: bool = True,
) -> Dict[str, Any]:
    try:
        body_bytes = await fastapi_request.body()
        req_size = len(body_bytes) if body_bytes else 0
    except Exception:
        req_size = 0

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            request_monitor.metrics.request_size_bytes = int(req_size)
        except Exception:
            pass

    if not request.prompt or not request.prompt.strip():
        if request_monitor and getattr(request_monitor, "metrics", None):
            try:
                request_monitor.metrics.error = True
                request_monitor.metrics.error_message = "Prompt cannot be empty"
                request_monitor.metrics.error_type = "ValidationError"
                request_monitor.metrics.status_code = 400
            except Exception:
                pass
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    try:
        req_max_tokens = int(request.max_tokens or 150)
        if req_max_tokens < 1:
            req_max_tokens = 1
        if req_max_tokens > MAX_TOKENS_LIMIT:
            req_max_tokens = MAX_TOKENS_LIMIT
    except Exception:
        req_max_tokens = 150

    if request.max_tokens and (
        request.max_tokens < 1 or request.max_tokens > MAX_TOKENS_LIMIT
    ):
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens must be between 1 and {MAX_TOKENS_LIMIT}",
        )
    if request.temperature and (request.temperature < 0 or request.temperature > 2):
        raise HTTPException(
            status_code=400, detail="temperature must be between 0 and 2"
        )

    final_prompt, was_truncated, trunc_reason = build_prompt_and_truncate(
        request.prompt, max_chars=DEFAULT_PROMPT_MAX_CHARS
    )

    chosen_model = (
        request.model.strip()
        if isinstance(request.model, str) and request.model and request.model.strip()
        else None
    ) or AZURE_OPENAI_DEPLOYMENT

    provider_override = None
    if isinstance(request.provider, str) and request.provider.strip():
        provider_override = request.provider.strip().lower()
        if provider_override not in ("azure", "ollama"):
            logger.warning("Unknown provider override received: %s", provider_override)

    token_est = {"plain_input_tokens": 0, "message_input_tokens": 0}
    try:
        token_est["plain_input_tokens"] = int(
            token_counter.count_tokens(final_prompt, None)
        )
    except Exception:
        token_est["plain_input_tokens"] = max(1, len(final_prompt) // 4)

    if do_quota_check:
        try:
            token_est["message_input_tokens"] = int(
                token_counter.count_message_tokens(
                    [{"role": "user", "content": final_prompt}], chosen_model
                )
            )
        except Exception:
            token_est["message_input_tokens"] = token_est["plain_input_tokens"]
    else:
        token_est["message_input_tokens"] = token_est["plain_input_tokens"]

    message_input_tokens = int(token_est["message_input_tokens"])

    if message_input_tokens > MAX_PROMPT_TOKENS:
        if request_monitor and getattr(request_monitor, "metrics", None):
            try:
                request_monitor.metrics.error = True
                request_monitor.metrics.error_message = f"Prompt too long: {message_input_tokens} tokens (limit {MAX_PROMPT_TOKENS})"
                request_monitor.metrics.error_type = "PromptTooLongError"
                request_monitor.metrics.status_code = 400
                setattr(
                    request_monitor.metrics,
                    "plain_input_tokens",
                    int(token_est["plain_input_tokens"]),
                )
                setattr(
                    request_monitor.metrics,
                    "message_input_tokens",
                    int(message_input_tokens),
                )
            except Exception:
                pass
        raise HTTPException(
            status_code=400,
            detail=f"Prompt too long: {message_input_tokens} tokens (limit is {MAX_PROMPT_TOKENS}).",
        )

    if request_monitor and request_monitor.sample and request_monitor.metrics:
        try:
            request_monitor.metrics.input_tokens = int(message_input_tokens)
            request_monitor.metrics.prompt_truncated = bool(was_truncated)
            setattr(
                request_monitor.metrics,
                "plain_input_tokens",
                int(token_est["plain_input_tokens"]),
            )
            setattr(
                request_monitor.metrics,
                "message_input_tokens",
                int(message_input_tokens),
            )
        except Exception:
            pass

    server_user_id = None
    if current_user:
        server_user_id = current_user.get("id")

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            request_monitor.metrics.user_id = server_user_id
        except Exception:
            pass

    user_hash = None
    try:
        if server_user_id:
            user_hash = hashlib.sha256(str(server_user_id).encode("utf-8")).hexdigest()[
                :8
            ]
            if request_monitor and getattr(request_monitor, "metrics", None):
                request_monitor.metrics.user_hash = user_hash
    except Exception:
        user_hash = None

    return {
        "formatted_prompt": final_prompt,
        "max_tokens": req_max_tokens,
        "temperature": request.temperature or 0.7,
        "user_id": server_user_id,
        "session_id": request.session_id,
        "was_truncated": was_truncated,
        "trunc_reason": trunc_reason,
        "input_tokens_est": message_input_tokens,
        "chosen_model": chosen_model,
        "provider": provider_override,
    }


async def check_and_increment_quota(processed_data: Dict[str, Any], request_monitor):
    try:
        server_user_id = processed_data.get("user_id")
        quota_user_id = (
            str(server_user_id)
            if server_user_id
            else (processed_data.get("user_hash") or "anonymous")
        )
    except Exception:
        quota_user_id = "anonymous"

    logger.debug("check_and_increment_quota called for quota_user=%s", quota_user_id)

    try:
        in_cd, seconds_left = await is_in_cooldown(quota_user_id)
        if in_cd:
            if request_monitor and getattr(request_monitor, "metrics", None):
                try:
                    request_monitor.metrics.error = True
                    request_monitor.metrics.error_message = "Quota cooldown active"
                    request_monitor.metrics.error_type = "QuotaCooldown"
                    request_monitor.metrics.status_code = 429
                except Exception:
                    pass
            raise HTTPException(
                status_code=429,
                detail=f"Quota exceeded. Try again in {seconds_left} seconds.",
                headers={"Retry-After": str(seconds_left)},
            )

        exceeded, current_count = await increment_and_check(quota_user_id, increment=1)
        if exceeded:
            await set_cooldown(quota_user_id)
            if request_monitor and request_monitor.metrics:
                try:
                    request_monitor.metrics.error = True
                    request_monitor.metrics.error_message = (
                        f"Quota exceeded ({current_count})"
                    )
                    request_monitor.metrics.error_type = "QuotaExceeded"
                    request_monitor.metrics.status_code = 429
                except Exception:
                    pass
            raise HTTPException(
                status_code=429,
                detail=f"Quota exceeded. Cooldown started.",
                headers={"Retry-After": str(settings.QUOTA_COOLDOWN_SECONDS)},
            )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Quota subsystem failed; allowing request")


@step("cache_lookup")
async def check_cache(cache_key: str, request_monitor) -> Optional[Dict[str, Any]]:
    cache_key_hash = None
    try:
        cache_key_hash = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
        if request_monitor and getattr(request_monitor, "metrics", None):
            request_monitor.metrics.cache_key_hash = cache_key_hash
    except Exception:
        cache_key_hash = None

    try:
        logger.debug(
            "CACHE GET - cache_key=%s cache_key_hash=%s", cache_key, cache_key_hash
        )
    except Exception:
        pass

    cache_start = time.perf_counter()
    try:
        raw = await cache_get_key(cache_key)
    except Exception as e:
        logger.exception("cache_get_key raised exception for key=%s: %s", cache_key, e)
        raw = None
    cache_end = time.perf_counter()
    cache_lookup_ms = (cache_end - cache_start) * 1000.0

    try:
        logger.debug(
            "RAW CACHE VALUE for key=%s type=%s repr_preview=%s",
            cache_key,
            type(raw).__name__,
            (str(raw)[:2000] if raw else "None"),
        )
    except Exception:
        pass

    cached_value = None
    try:
        if raw is None:
            cached_value = None
        elif isinstance(raw, bytes):
            try:
                cached_value = json.loads(raw.decode("utf-8"))
            except Exception:
                cached_value = None
        elif isinstance(raw, str):
            try:
                cached_value = json.loads(raw)
            except Exception:
                cached_value = None
        elif isinstance(raw, dict):
            if "value" in raw and isinstance(raw["value"], (dict, str)):
                val = raw["value"]
                if isinstance(val, str):
                    try:
                        cached_value = json.loads(val)
                    except Exception:
                        cached_value = val if isinstance(val, dict) else None
                else:
                    cached_value = val
            else:
                cached_value = raw
        else:
            val = getattr(raw, "value", None)
            if isinstance(val, (str, bytes)):
                try:
                    cached_value = json.loads(
                        val.decode("utf-8") if isinstance(val, bytes) else val
                    )
                except Exception:
                    cached_value = None
            elif isinstance(val, dict):
                cached_value = val
            else:
                cached_value = None
    except Exception:
        cached_value = None

    if cached_value:
        logger.info(
            "CACHE HIT key_hash=%s lookup_ms=%.2f raw_type=%s",
            cache_key_hash,
            cache_lookup_ms,
            type(raw).__name__,
        )
        if request_monitor and getattr(request_monitor, "metrics", None):
            try:
                request_monitor.metrics.cache_hit = True
                request_monitor.metrics.cache_lookup_ms = float(cache_lookup_ms)
            except Exception:
                pass
    else:
        logger.info(
            "CACHE MISS key_hash=%s lookup_ms=%.2f raw_type=%s",
            cache_key_hash,
            cache_lookup_ms,
            type(raw).__name__,
        )

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            request_monitor.metrics.cache_lookup_ms = float(cache_lookup_ms)
        except Exception:
            pass

    return cached_value


async def _call_azure_llm(
    client, model: str, prompt: str, max_tokens: int, temperature: float
):
    if client is None:
        raise RuntimeError("Azure client not configured")
    return await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def _call_ollama_llm(
    client, model: str, prompt: str, max_tokens: int, temperature: float
):
    messages = [{"role": "user", "content": prompt}]
    http_payload = {
        "model": model,
        "prompt": prompt,
        "max_new_tokens": max_tokens,
        "temperature": float(temperature),
        "stream": False,
    }
    if client is not None:
        try_calls = []
        try_calls.append(
            (
                "chat",
                {
                    "model": model,
                    "messages": messages,
                    "temperature": float(temperature),
                },
            )
        )
        try_calls.append(
            ("chat", {"messages": messages, "temperature": float(temperature)})
        )
        try_calls.append(
            (
                "chat",
                {
                    "model": model,
                    "messages": messages,
                    "max_new_tokens": max_tokens,
                    "temperature": float(temperature),
                },
            )
        )
        try_calls.append(
            (
                "chat",
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": float(temperature),
                },
            )
        )
        try_calls.append(
            (
                "generate",
                {
                    "model": model,
                    "prompt": prompt,
                    "max_new_tokens": max_tokens,
                    "temperature": float(temperature),
                },
            )
        )
        try_calls.append(
            (
                "generate",
                {
                    "prompt": prompt,
                    "max_new_tokens": max_tokens,
                    "temperature": float(temperature),
                },
            )
        )
        for fn_name, kwargs in try_calls:
            if not hasattr(client, fn_name):
                continue
            fn = getattr(client, fn_name)
            try:
                maybe = fn(**kwargs)
                if asyncio.iscoroutine(maybe):
                    maybe = await maybe
                return maybe
            except TypeError as te:
                logger.debug(
                    "Ollama client.%s signature TypeError: %s — trying next signature",
                    fn_name,
                    te,
                )
                continue
            except Exception as e:
                logger.warning(
                    "Ollama client.%s call raised: %s — falling back to HTTP if available",
                    fn_name,
                    e,
                )
                break

    if httpx is None:
        raise RuntimeError(
            "Ollama client failed and httpx is not installed for HTTP fallback. Install httpx or fix the client usage."
        )

    async with httpx.AsyncClient(timeout=PER_ATTEMPT_TIMEOUT) as httpc:
        url = f"{OLLAMA_API_URL.rstrip('/')}/api/generate"
        try:
            resp = await httpc.post(url, json=http_payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as he:
            raise RuntimeError(
                f"Ollama HTTP API error: {he.response.status_code} {he.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Ollama HTTP request failed: {e}")


@step("llm_api_call")
async def call_llm_api(processed_data: Dict[str, Any]) -> Dict[str, Any]:
    chosen_model = processed_data.get("chosen_model")
    provider = processed_data.get("provider") or _select_provider_for_model(
        chosen_model
    )

    last_exc = None
    sdk_response = None
    total_llm_time = 0.0

    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempt_start = time.perf_counter()
        attempt_time = 0.0
        try:
            logger.info(
                f"LLM API call attempt {attempt}/{MAX_ATTEMPTS} (model={chosen_model} provider={provider})"
            )
            if provider == "azure":
                sdk_response = await asyncio.wait_for(
                    _call_azure_llm(
                        azure_client,
                        chosen_model,
                        processed_data["formatted_prompt"],
                        processed_data["max_tokens"],
                        processed_data["temperature"],
                    ),
                    timeout=PER_ATTEMPT_TIMEOUT,
                )
            elif provider == "ollama":
                sdk_response = await asyncio.wait_for(
                    _call_ollama_llm(
                        ollama_client,
                        chosen_model,
                        processed_data["formatted_prompt"],
                        processed_data["max_tokens"],
                        processed_data["temperature"],
                    ),
                    timeout=PER_ATTEMPT_TIMEOUT,
                )
            else:
                raise RuntimeError(f"Unknown provider for model `{chosen_model}`.")
            attempt_time = (time.perf_counter() - attempt_start) * 1000.0
            total_llm_time += attempt_time
            break
        except asyncio.TimeoutError as e:
            last_exc = e
            logger.warning(f"LLM API call timed out on attempt {attempt}")
        except Exception as e:
            last_exc = e
            logger.warning(f"LLM API call attempt {attempt} failed: {e}")
        finally:
            try:
                total_llm_time += attempt_time
                logger.debug(f"LLM attempt {attempt} elapsed {attempt_time:.3f}ms")
            except Exception:
                pass

        if attempt < MAX_ATTEMPTS:
            backoff = BASE_BACKOFF * (2 ** (attempt - 1))
            jitter = (time.time() % 0.5) * 0.1
            await asyncio.sleep(backoff + jitter)

    if sdk_response is None:
        logger.exception("LLM API calls failed after retries")
        if isinstance(last_exc, asyncio.TimeoutError):
            raise HTTPException(status_code=504, detail="LLM API call timed out.")
        else:
            raise HTTPException(
                status_code=502, detail=f"LLM API call failed: {str(last_exc)}"
            )

    return {
        "sdk_response": sdk_response,
        "total_llm_time_ms": total_llm_time,
        "attempts": attempt,
    }


@step("postprocessing")
async def process_response(
    llm_result: Dict[str, Any], processed_data: Dict[str, Any], request_monitor
) -> Dict[str, Any]:
    try:
        token_count_start = time.perf_counter()
        (
            content,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        ) = _extract_response_and_usage(
            llm_result["sdk_response"],
            processed_data["chosen_model"],
            processed_data["input_tokens_est"],
        )
        token_count_end = time.perf_counter()
        token_count_ms = (token_count_end - token_count_start) * 1000.0
    except Exception as e:
        logger.exception("Failed to parse LLM SDK response")
        raise HTTPException(status_code=500, detail="Failed to parse LLM response")

    try:
        logger.debug("LLM extracted content preview=%s", (content or "")[:1000])
    except Exception:
        pass

    if not (content and str(content).strip()):
        logger.warning(
            "LLM extracted empty content for model=%s; prompt_tokens=%s completion_tokens=%s",
            processed_data.get("chosen_model"),
            prompt_tokens,
            completion_tokens,
        )

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            setattr(
                request_monitor.metrics,
                "postprocessing_token_count_ms",
                float(token_count_ms),
            )
        except Exception:
            pass
        try:
            steps = getattr(request_monitor.metrics, "steps", {}) or {}
            if not isinstance(steps, dict):
                steps = {}
            steps["postprocessing_token_count_ms"] = float(token_count_ms)
            request_monitor.metrics.steps = steps
        except Exception:
            pass

    if request_monitor:
        try:
            request_monitor.update_tokens(
                input_tokens=prompt_tokens, output_tokens=completion_tokens
            )
        except Exception:
            pass
        try:
            request_monitor.update_response(content, status_code=200)
        except Exception:
            pass

    try:
        cost_start = time.perf_counter()
        estimated_cost = cost_calculator.calculate_cost(
            processed_data.get("chosen_model"),
            int(prompt_tokens),
            int(completion_tokens),
            request_monitor=request_monitor,
            context=processed_data,
        )
        cost_end = time.perf_counter()
        cost_ms = (cost_end - cost_start) * 1000.0
    except Exception:
        estimated_cost = 0.0
        cost_ms = 0.0

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            setattr(
                request_monitor.metrics, "postprocessing_cost_calc_ms", float(cost_ms)
            )
        except Exception:
            pass
        try:
            steps = getattr(request_monitor.metrics, "steps", {}) or {}
            if not isinstance(steps, dict):
                steps = {}
            steps["postprocessing_cost_calc_ms"] = float(cost_ms)
            request_monitor.metrics.steps = steps
        except Exception:
            pass

    try:
        dec_cost = Decimal(str(estimated_cost or 0)).quantize(
            Decimal("0.00000001"), rounding=ROUND_DOWN
        )
    except Exception:
        dec_cost = Decimal("0.0")
    est_cost_float = float(dec_cost)
    est_cost_str = format(dec_cost, "f")

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            request_monitor.metrics.input_tokens = int(prompt_tokens)
            request_monitor.metrics.output_tokens = int(completion_tokens)
            request_monitor.metrics.total_tokens = int(total_tokens)
            try:
                request_monitor.metrics.estimated_cost_usd_decimal = dec_cost
            except Exception:
                pass
            try:
                request_monitor.metrics.estimated_cost_usd = est_cost_float
            except Exception:
                request_monitor.metrics.estimated_cost_usd = float(est_cost_float)
            try:
                request_monitor.metrics.estimated_cost_usd_str = est_cost_str
            except Exception:
                pass
        except Exception:
            pass

    resp_text = content or ""
    try:
        resp_size = len(resp_text.encode("utf-8"))
    except Exception:
        resp_size = 0

    if request_monitor and getattr(request_monitor, "metrics", None):
        try:
            request_monitor.metrics.response_size_bytes = int(resp_size)
        except Exception:
            pass

    return {
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost": est_cost_float,
        "estimated_cost_str": est_cost_str,
        "estimated_cost_decimal": dec_cost,
        "response_size": resp_size,
    }


@step("metrics_export")
async def export_metrics_and_cache(
    response_data: Dict[str, Any],
    processed_data: Dict[str, Any],
    cache_key: str,
    request_monitor,
    experiment: str,
) -> Dict[str, Any]:
    """
    Exports metrics and always writes a JSON payload to cache with response + usage + metrics.
    """
    if request_monitor:
        current_metrics = request_monitor.get_current_metrics()
    else:
        current_metrics = {}

    if current_metrics is None:
        current_metrics = {}

    try:
        if request_monitor and getattr(request_monitor, "metrics", None):
            if getattr(request_monitor.metrics, "error", False):
                logger.warning(
                    "Not caching response because request_monitor indicates error for request %s",
                    getattr(request_monitor.metrics, "request_id", "unknown"),
                )
                return current_metrics
    except Exception:
        pass

    try:
        current_metrics.setdefault("user_id", processed_data.get("user_id"))
    except Exception:
        pass

    try:
        steps = (
            current_metrics.get("steps", {})
            if isinstance(current_metrics, dict)
            else {}
        )
        current_metrics["steps"] = steps
        current_metrics["tokens_used"] = int(response_data.get("total_tokens", 0))

        est_cost_str = response_data.get("estimated_cost_str")
        if not est_cost_str:
            try:
                dec_cost = Decimal(
                    str(response_data.get("estimated_cost", 0.0))
                ).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
                est_cost_str = format(dec_cost, "f")
            except Exception:
                est_cost_str = "0.0"

        try:
            dec_cost_val = response_data.get("estimated_cost_decimal")
            if dec_cost_val is None:
                dec_cost_val = Decimal(est_cost_str)
        except Exception:
            try:
                dec_cost_val = Decimal(str(response_data.get("estimated_cost", 0.0)))
            except Exception:
                dec_cost_val = Decimal("0.0")

        try:
            dec_cost_val = dec_cost_val.quantize(
                Decimal("0.00000001"), rounding=ROUND_DOWN
            )
        except Exception:
            pass

        try:
            dec_cost_float = float(dec_cost_val)
        except Exception:
            dec_cost_float = 0.0

        current_metrics["estimated_cost_usd_str"] = est_cost_str
        current_metrics["estimated_cost_usd"] = dec_cost_float

    except Exception:
        logger.debug("Failed to enrich exported metrics", exc_info=True)

    try:
        cache_hit_flag = False
        if request_monitor and getattr(request_monitor.metrics, "cache_hit", False):
            cache_hit_flag = True

        if not cache_hit_flag:
            current_metrics.setdefault("cache_hit", False)
            if hasattr(request_monitor.metrics, "cache_lookup_ms"):
                current_metrics.setdefault(
                    "cache_lookup_ms", request_monitor.metrics.cache_lookup_ms
                )
            uh = processed_data.get("user_hash")
            ckh = (
                getattr(request_monitor.metrics, "cache_key_hash", None)
                if request_monitor
                else None
            )
            if uh:
                current_metrics.setdefault("user_hash", uh)
            if ckh:
                current_metrics.setdefault("cache_key_hash", ckh)
    except Exception:
        pass

    try:
        cache_key_hash_val = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:12]
    except Exception:
        cache_key_hash_val = processed_data.get("user_hash", "unknown")

    resp_text = response_data.get("content") or ""
    cache_payload = {
        "response": resp_text,
        "usage": {
            "prompt_tokens": int(response_data.get("prompt_tokens", 0)),
            "completion_tokens": int(response_data.get("completion_tokens", 0)),
            "total_tokens": int(response_data.get("total_tokens", 0)),
        },
        "model": processed_data.get("chosen_model"),
        "cached_at": time.time(),
        "metrics": current_metrics,
        "cache_key_hash": cache_key_hash_val,
        "estimated_cost_str": response_data.get("estimated_cost_str"),
        "estimated_cost_float": float(current_metrics.get("estimated_cost_usd", 0.0)),
    }

    try:
        payload_json = json.dumps(cache_payload, default=str, ensure_ascii=False)
        logger.debug(
            "ABOUT TO WRITE CACHE key=%s cache_key_hash=%s ttl=%s payload_preview=%s",
            cache_key,
            cache_key_hash_val,
            CACHE_TTL,
            payload_json[:400],
        )
        ok = await cache_set_key(cache_key, payload_json, ttl_seconds=CACHE_TTL)
        if ok:
            logger.info(
                "CACHE SET OK cache_key=%s key_hash=%s ttl=%s",
                cache_key,
                cache_key_hash_val,
                CACHE_TTL,
            )
        else:
            logger.warning(
                "CACHE SET FAILED (returned False) cache_key=%s key_hash=%s",
                cache_key,
                cache_key_hash_val,
            )
    except Exception as e:
        logger.exception(
            "Cache set exception for cache_key=%s key_hash=%s: %s",
            cache_key,
            cache_key_hash_val,
            e,
        )

    if processed_data.get("was_truncated"):
        current_metrics["prompt_truncated"] = True
        current_metrics["truncation_reason"] = processed_data.get("trunc_reason")

    return current_metrics


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(
    request: GenerateRequest,
    fastapi_request: Request,
    current_user: dict = Depends(get_current_user),
):
    request_monitor = getattr(fastapi_request.state, "llm_monitor", None)
    experiment = getattr(fastapi_request.state, "experiment", None) or decide_variant(
        fastapi_request
    )

    try:
        processed_data = await preprocess_request(
            request,
            fastapi_request,
            current_user,
            request_monitor,
            do_quota_check=False,
        )

        try:
            if request_monitor and getattr(request_monitor, "metrics", None):
                request_monitor.metrics.model = (
                    processed_data.get("chosen_model") or request_monitor.metrics.model
                )
        except Exception:
            pass

        _prompt_for_cache = _canonicalize_prompt(
            processed_data.get("formatted_prompt") or ""
        )
        _provider_norm, _model_norm = _normalize_provider_model(
            processed_data.get("provider"), processed_data.get("chosen_model")
        )

        cache_opts = {
            "model": _model_norm or "",
            "provider": _provider_norm or "",
            "max_tokens": int(request.max_tokens or 0),
            "temperature": float(request.temperature or 0.0),
            "experiment": experiment or "",
        }

        logger.debug(
            "Build cache key - prompt_preview=%s opts=%s",
            _prompt_for_cache[:200],
            cache_opts,
        )
        cache_key = cache_make_key(_prompt_for_cache, cache_opts)

        cached_value = await check_cache(cache_key, request_monitor)
        if cached_value:
            exporter_inst = get_global_exporter()
            if exporter_inst:
                try:
                    exporter_inst.cache_hits.labels(
                        endpoint="/generate", experiment=experiment
                    ).inc()
                except Exception:
                    pass

            if not isinstance(cached_value, dict):
                logger.warning(
                    "Cached value is not dict; ignoring cache. key=%s val_type=%s",
                    cache_key,
                    type(cached_value).__name__,
                )
            else:
                logger.info(
                    "CACHE HIT - Using pre-computed values, skipping all token/cost calculations"
                )
                usage = cached_value.get("usage", {}) or {}
                cached_metrics = cached_value.get("metrics", {}) or {}
                input_tokens_cached = int(usage.get("prompt_tokens", 0))
                output_tokens_cached = int(usage.get("completion_tokens", 0))
                total_cached = int(
                    usage.get(
                        "total_tokens", input_tokens_cached + output_tokens_cached
                    )
                )
                cached_cost_str = cached_value.get("estimated_cost_str", "0.0")
                try:
                    dec_cached = Decimal(str(cached_cost_str))
                    cached_cost_float = float(dec_cached)
                except Exception:
                    cached_cost_float = 0.0
                    dec_cached = Decimal("0.0")

                if (
                    request_monitor
                    and request_monitor.sample
                    and request_monitor.metrics
                ):
                    try:
                        request_monitor.metrics.cache_hit = True
                        cached_model = cached_value.get("model")
                        if cached_model:
                            request_monitor.metrics.model = str(cached_model)
                        request_monitor.metrics.input_tokens = int(input_tokens_cached)
                        request_monitor.metrics.output_tokens = int(
                            output_tokens_cached
                        )
                        request_monitor.metrics.total_tokens = int(total_cached)
                        try:
                            request_monitor.metrics.estimated_cost_usd_decimal = (
                                dec_cached
                            )
                        except Exception:
                            pass
                        try:
                            request_monitor.metrics.estimated_cost_usd = (
                                cached_cost_float
                            )
                        except Exception:
                            request_monitor.metrics.estimated_cost_usd = float(
                                cached_cost_float
                            )
                        try:
                            request_monitor.metrics.estimated_cost_usd_str = (
                                cached_cost_str
                            )
                        except Exception:
                            pass

                        request_monitor.update_response(
                            cached_value.get("response", ""), status_code=200
                        )
                    except Exception:
                        logger.exception("Failed updating monitor for cache hit")

                if isinstance(cached_metrics, dict):
                    cached_metrics["cache_hit"] = True
                    cached_metrics.setdefault("tokens_used", total_cached)
                    try:
                        if isinstance(
                            cached_metrics.get("estimated_cost_usd"), Decimal
                        ):
                            cached_metrics["estimated_cost_usd"] = float(
                                cached_metrics["estimated_cost_usd"]
                            )
                    except Exception:
                        pass
                    cached_metrics.setdefault("estimated_cost_usd_str", cached_cost_str)
                    try:
                        cached_metrics.setdefault(
                            "estimated_cost_usd", float(cached_cost_float)
                        )
                    except Exception:
                        cached_metrics.setdefault("estimated_cost_usd", float(0.0))
                    if processed_data.get("user_hash"):
                        cached_metrics.setdefault(
                            "user_hash", processed_data["user_hash"]
                        )
                    for f in ("error", "error_message", "error_type", "status_code"):
                        cached_metrics.pop(f, None)

                logger.info(
                    "Returning cached response for key_hash=%s with pre-computed tokens=%d, cost=%s",
                    getattr(request_monitor.metrics, "cache_key_hash", cache_key)
                    if request_monitor and getattr(request_monitor, "metrics", None)
                    else cache_key,
                    total_cached,
                    cached_cost_str,
                )

                return GenerateResponse(
                    response=cached_value.get("response", ""), metrics=cached_metrics
                )

        exporter_inst = get_global_exporter()
        if exporter_inst:
            try:
                exporter_inst.cache_misses.labels(
                    endpoint="/generate", experiment=experiment
                ).inc()
            except Exception:
                pass

        logger.info(
            "CACHE MISS - Will perform full LLM call with token/cost calculations"
        )

        await check_and_increment_quota(processed_data, request_monitor)

        try:
            if request_monitor:
                request_monitor.start_step("llm_api_call")
            llm_result = await call_llm_api(processed_data)
        finally:
            try:
                if request_monitor:
                    request_monitor.stop_current_step()
            except Exception:
                pass

        try:
            fastapi_request.state.current_model = processed_data.get("chosen_model")
        except Exception:
            pass

        logger.info("Processing LLM response - calculating tokens and costs")
        try:
            if request_monitor:
                request_monitor.start_step("postprocessing")
            response_data = await process_response(
                llm_result, processed_data, request_monitor
            )
        finally:
            try:
                if request_monitor:
                    request_monitor.stop_current_step()
            except Exception:
                pass

        final_metrics = await export_metrics_and_cache(
            response_data, processed_data, cache_key, request_monitor, experiment
        )

        final_metrics.setdefault("cache_hit", False)

        logger.info(
            "LLM call completed - tokens=%d, cost=%s",
            response_data.get("total_tokens", 0),
            response_data.get("estimated_cost_str", "0.0"),
        )
        try:
            if request_monitor and getattr(request_monitor, "metrics", None):
                try:
                    if (
                        not getattr(request_monitor.metrics, "model", None)
                        or request_monitor.metrics.model == "unknown-model"
                    ):
                        request_monitor.metrics.model = processed_data.get(
                            "chosen_model"
                        )

                    cm = {}
                    try:
                        cm = request_monitor.get_current_metrics() or {}
                    except Exception:
                        cm = {}

                    steps = cm.get("steps", {}) if isinstance(cm, dict) else {}

                    try:
                        request_monitor.metrics.preprocessing_ms = float(
                            steps.get(
                                "preprocessing_ms",
                                getattr(
                                    request_monitor.metrics, "preprocessing_ms", 0.0
                                )
                                or 0.0,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        request_monitor.metrics.llm_api_call_ms = float(
                            steps.get(
                                "llm_api_call_ms",
                                getattr(request_monitor.metrics, "llm_api_call_ms", 0.0)
                                or 0.0,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        request_monitor.metrics.postprocessing_ms = float(
                            steps.get(
                                "postprocessing_ms",
                                getattr(
                                    request_monitor.metrics, "postprocessing_ms", 0.0
                                )
                                or 0.0,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        request_monitor.metrics.metrics_export_ms = float(
                            steps.get(
                                "metrics_export_ms",
                                getattr(
                                    request_monitor.metrics, "metrics_export_ms", 0.0
                                )
                                or 0.0,
                            )
                        )
                    except Exception:
                        pass
                    try:
                        request_monitor.metrics.total_latency_ms = float(
                            cm.get(
                                "total_latency_ms",
                                getattr(
                                    request_monitor.metrics, "total_latency_ms", 0.0
                                )
                                or 0.0,
                            )
                        )
                    except Exception:
                        pass

                except Exception:
                    logger.exception(
                        "Failed to enrich metrics object before persisting"
                    )

                logger.debug(
                    "Metrics before persist: %s",
                    getattr(
                        request_monitor.metrics, "__dict__", request_monitor.metrics
                    ),
                )
                await storage.store_metrics(request_monitor.metrics)
                logger.info("Metrics persisted to Postgres (async)")
        except Exception as e:
            logger.exception("Failed to persist metrics: %s", e)

        return GenerateResponse(
            response=response_data["content"], metrics=final_metrics
        )

    except HTTPException as he:
        if request_monitor and getattr(request_monitor, "metrics", None):
            try:
                request_monitor.metrics.error = True
                request_monitor.metrics.error_message = (
                    he.detail if hasattr(he, "detail") else str(he)
                )
                request_monitor.metrics.error_type = type(he).__name__
                request_monitor.metrics.status_code = int(
                    getattr(he, "status_code", 400)
                )
            except Exception:
                pass
        raise
    except Exception as e:
        if request_monitor and getattr(request_monitor, "metrics", None):
            try:
                request_monitor.metrics.error = True
                request_monitor.metrics.error_message = str(e)
                request_monitor.metrics.error_type = type(e).__name__
                request_monitor.metrics.status_code = int(
                    getattr(e, "status_code", 500)
                )
            except Exception:
                pass
        logger.exception("Unexpected error in /generate")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/")
async def root():
    return {
        "message": "LLM Application with Performance Monitoring",
        "version": "1.0.0",
        "endpoints": {
            "/generate": "Generate text",
            "/available_models": "GET model list per provider",
            "/metrics": "Prometheus metrics",
            "/health": "Health check",
        },
        "default_provider": DEFAULT_PROVIDER,
        "azure_default_model": AZURE_OPENAI_DEPLOYMENT,
    }


@router.get("/metrics")
async def metrics_protected():
    try:
        exporter = get_global_exporter()
        registry = getattr(exporter, "registry", None) if exporter else None
        payload = generate_latest() if registry is None else generate_latest(registry)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics generation failed: {e}")


def register_storage_events(app: FastAPI):
    @app.on_event("startup")
    async def _startup():
        await storage.init_pool()
        logger.info("AsyncPostgreSQLStorage initialized on startup")

    @app.on_event("shutdown")
    async def _shutdown():
        await storage.close()
        logger.info("AsyncPostgreSQLStorage closed on shutdown")
