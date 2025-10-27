# llm_apm/core/decorators.py
import functools
import logging
import inspect
import time
from typing import Any, Callable, Optional, Dict
from contextvars import ContextVar

from ..core.timer import Timer  

logger = logging.getLogger(__name__)

current_step_timer: ContextVar[Optional[dict]] = ContextVar('current_step_timer', default=None)

def _get_context() -> dict:
 
    ctx = current_step_timer.get()
    if ctx is None:
        ctx = {}
    return ctx

def step(step_name: str):
   
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            timer = Timer()
            start_ts = timer.start()
            context = _get_context()
            context_entry = {
                "start_time": float(start_ts),
                "running": True,
                "duration_ms": None,
                "success": None,
                "error": None
            }
            try:
                if current_step_timer.get() is not None:
                    ctx = current_step_timer.get()
                    ctx[step_name] = context_entry
                    current_step_timer.set(ctx)
                else:
                    context[step_name] = context_entry
            except Exception:
                try:
                    context[step_name] = context_entry
                except Exception:
                    pass

            try:
                result = func(*args, **kwargs)
                elapsed_ms = timer.stop() * 1000.0
                context_entry.update({
                    "duration_ms": float(elapsed_ms),
                    "running": False,
                    "success": True,
                    "error": None
                })

                try:
                    if current_step_timer.get() is not None:
                        ctx = current_step_timer.get()
                        ctx[step_name] = context_entry
                        current_step_timer.set(ctx)
                    else:
                        context[step_name] = context_entry
                except Exception:
                    pass

                logger.debug("Step '%s' completed in %.2fms", step_name, elapsed_ms)
                return result
            except Exception as e:
                try:
                    elapsed_ms = timer.stop() * 1000.0
                except Exception:
                    elapsed_ms = 0.0
                context_entry.update({
                    "duration_ms": float(elapsed_ms),
                    "running": False,
                    "success": False,
                    "error": str(e)
                })
                try:
                    if current_step_timer.get() is not None:
                        ctx = current_step_timer.get()
                        ctx[step_name] = context_entry
                        current_step_timer.set(ctx)
                    else:
                        context[step_name] = context_entry
                except Exception:
                    pass

                logger.error("Step '%s' failed after %.2fms: %s", step_name, elapsed_ms, e)
                raise

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            timer = Timer()
            start_ts = timer.start()
            context = _get_context()
            context_entry = {
                "start_time": float(start_ts),
                "running": True,
                "duration_ms": None,
                "success": None,
                "error": None
            }
            try:
                if current_step_timer.get() is not None:
                    ctx = current_step_timer.get()
                    ctx[step_name] = context_entry
                    current_step_timer.set(ctx)
                else:
                    context[step_name] = context_entry
            except Exception:
                try:
                    context[step_name] = context_entry
                except Exception:
                    pass

            try:
                result = await func(*args, **kwargs)
                elapsed_ms = timer.stop() * 1000.0
                context_entry.update({
                    "duration_ms": float(elapsed_ms),
                    "running": False,
                    "success": True,
                    "error": None
                })
                try:
                    if current_step_timer.get() is not None:
                        ctx = current_step_timer.get()
                        ctx[step_name] = context_entry
                        current_step_timer.set(ctx)
                    else:
                        context[step_name] = context_entry
                except Exception:
                    pass

                logger.debug("Step '%s' completed in %.2fms", step_name, elapsed_ms)
                return result
            except Exception as e:
                try:
                    elapsed_ms = timer.stop() * 1000.0
                except Exception:
                    elapsed_ms = 0.0
                context_entry.update({
                    "duration_ms": float(elapsed_ms),
                    "running": False,
                    "success": False,
                    "error": str(e)
                })
                try:
                    if current_step_timer.get() is not None:
                        ctx = current_step_timer.get()
                        ctx[step_name] = context_entry
                        current_step_timer.set(ctx)
                    else:
                        context[step_name] = context_entry
                except Exception:
                    pass

                logger.error("Step '%s' failed after %.2fms: %s", step_name, elapsed_ms, e)
                raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator

def get_current_step_metrics() -> Dict[str, float]:
    """
    Read the current ContextVar dict and return a map of { '<step>_ms': <float> }.
    For running steps we compute a best-effort elapsed using stored start_time.
    """
    ctx = current_step_timer.get()
    metrics: Dict[str, float] = {}
    now = time.perf_counter()
    if not ctx:
        logger.debug("get_current_step_metrics: no current_step_timer context found")
        return metrics

    for step_name, data in list(ctx.items()):
        try:
            if not isinstance(data, dict):
                continue
            if data.get("duration_ms") is not None:
                metrics[f"{step_name}_ms"] = float(data["duration_ms"])
            else:
                start = data.get("start_time")
                if start:
                    try:
                        elapsed_ms = (now - float(start)) * 1000.0
                        metrics[f"{step_name}_ms"] = float(elapsed_ms)
                    except Exception:
                        metrics[f"{step_name}_ms"] = 0.0
                else:
                    metrics[f"{step_name}_ms"] = 0.0
        except Exception:
            metrics[f"{step_name}_ms"] = 0.0
    return metrics

def clear_step_context():
    
    try:
        current_step_timer.set({})
    except Exception:
        pass
