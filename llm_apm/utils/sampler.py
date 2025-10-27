
from typing import Optional, Dict
import hashlib
import logging

logger = logging.getLogger(__name__)


def _hash_to_fraction(key: str) -> float:
    """
    Hash a string to a deterministic fraction in [0.0, 1.0).
    Uses SHA-256 and converts the first 8 bytes to an int to keep behavior stable.
    """
    if not key:
        return 0.0
    h = hashlib.sha256(key.encode("utf-8")).digest()
    val = int.from_bytes(h[:8], "big")
    return (val % (10**9)) / 10**9


def should_sample(user_id: Optional[str], sampling_rate: float) -> bool:
    
    if sampling_rate >= 1.0:
        return True
    if user_id:
        frac = _hash_to_fraction(user_id)
        return frac < sampling_rate
    import random
    return random.random() < sampling_rate


def decide_variant(user_id: Optional[str], variants: Optional[Dict[str, float]] = None) -> str:
    
    if not variants:
        return "control"
    items = list(variants.items())
    cumulative = []
    total = 0.0
    for name, weight in items:
        total += float(weight)
        cumulative.append((name, total))
    frac = _hash_to_fraction(user_id) if user_id else _hash_to_fraction("random_fallback")
    for name, bound in cumulative:
        if frac < bound:
            return name
    return items[-1][0]


def get_sampling_rate() -> float:
    """Return sampling rate from config (safe fallback to 1.0)."""
    try:
        from llm_apm.config.settings import config
        return float(getattr(config, "sampling_rate", 1.0))
    except Exception as e:
        logger.debug(f"Could not read sampling_rate from config: {e}")
        return 1.0
