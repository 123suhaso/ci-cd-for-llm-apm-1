from typing import Optional, List, Dict, Any
import logging
import re

logger = logging.getLogger(__name__)


MODEL_CHAT_RULES = {
    "gpt-3.5-turbo":        (4, -1, 2),
    "gpt-3.5-turbo-0301":   (4, -1, 2),
    "gpt-4":                (3, 1, 3),
    "gpt-4o":               (3, 1, 3),
    "gpt-4o-mini":          (3, 1, 3),
    "gpt-4-turbo":          (3, 1, 3),
    "default":              (4, 0, 2),
}

class _FallbackEncoder:
    def __init__(self, chars_per_token: float = 3.3):
        self.chars_per_token = chars_per_token

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        s = str(text).strip()
        if not s:
            return []
        spaces_punct = len(re.findall(r'[\s.,!?;:()"\-]', s))
        common_words = len(re.findall(r'\b(?:the|and|for|are|but|not|you|all|can|was|one|now|new|use|what|when|will|with)\b', s.lower()))
        eff = max(1, len(s) - int(min(len(s)*0.25, (spaces_punct + common_words) * 0.5)))
        est = max(1, int(eff / self.chars_per_token))
        return list(range(est))

class TokenCounter:
    def __init__(self):
        self._tiktoken = None
        self._enc_cache: Dict[str, Any] = {}
        self._tiktoken_available: Optional[bool] = None

    def _load_tiktoken(self) -> bool:
        if self._tiktoken_available is not None:
            return self._tiktoken_available
        try:
            import tiktoken
            self._tiktoken = tiktoken
            self._tiktoken_available = True
            logger.info("tiktoken available — using it for exact counts")
        except ImportError:
            self._tiktoken = None
            self._tiktoken_available = False
            logger.info("tiktoken NOT installed — using fallback heuristic (counts will differ from OpenAI tokenizer)")
        return self._tiktoken_available

    def _get_encoder(self, model: Optional[str] = None):
        key = (model or "__default__").lower()
        if key in self._enc_cache:
            return self._enc_cache[key]

        if not self._load_tiktoken():
            enc = _FallbackEncoder(chars_per_token=3.3)
            self._enc_cache[key] = enc
            return enc

        try:
            try:
                enc = self._tiktoken.encoding_for_model(model) if model else self._tiktoken.get_encoding("cl100k_base")
            except Exception:
                enc = self._tiktoken.get_encoding("cl100k_base")
            self._enc_cache[key] = enc
            return enc
        except Exception as e:
            logger.warning("tiktoken encoder error for %s: %s", model, e)
            enc = _FallbackEncoder()
            self._enc_cache[key] = enc
            return enc

    def _get_model_rules(self, model: Optional[str]):
        if not model:
            return MODEL_CHAT_RULES["default"]
        m = model.lower()
        if m in MODEL_CHAT_RULES:
            return MODEL_CHAT_RULES[m]
        for k in MODEL_CHAT_RULES:
            if k != "default" and k in m:
                return MODEL_CHAT_RULES[k]
        return MODEL_CHAT_RULES["default"]

    def count_tokens(self, text: Optional[str], model: Optional[str] = None) -> int:
        """Count tokens in a plain string using encoder (tiktoken if available)."""
        if not text:
            return 0
        enc = self._get_encoder(model)
        try:
            toks = enc.encode(str(text))
            return len(toks)
        except Exception:
            s = str(text).strip()
            if not s:
                return 0
            return max(1, int(len(s) / 3.3))

    def count_message_tokens(self, messages: List[Dict[str, Any]], model: Optional[str] = None) -> int:
        """
        Count tokens for a chat-style messages list (each message is a dict
        with at least 'role' and 'content', optionally 'name').
        Implements OpenAI-style message overhead rules.
        """
        if not messages:
            return 0
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")

        enc = self._get_encoder(model)
        per_msg, per_name_delta, reply_overhead = self._get_model_rules(model)
        total = 0

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            total += per_msg

            content = msg.get("content", "")
            if content:
                try:
                    total += len(enc.encode(str(content)))
                except Exception:
                    total += max(1, int(len(str(content).strip()) / 3.3))

            name = msg.get("name")
            role = msg.get("role")
            if name:
                try:
                    total += len(enc.encode(str(name)))
                except Exception:
                    total += max(1, int(len(str(name).strip()) / 3.3))
                total += int(per_name_delta)
            else:
                if role:
                    try:
                        total += len(enc.encode(str(role)))
                    except Exception:
                        total += max(1, int(len(str(role).strip()) / 3.3))

            for k, v in msg.items():
                if k in ("role", "name", "content"):
                    continue
                if isinstance(v, str) and v:
                    try:
                        total += len(enc.encode(v))
                    except Exception:
                        total += max(1, int(len(v.strip()) / 3.3))

        total += reply_overhead
        return int(total)

    def explain_tokenization(self, text: str, model: Optional[str] = None) -> Dict[str, Any]:
        enc = self._get_encoder(model)
        try:
            toks = enc.encode(text)
            return {
                "token_count": len(toks),
                "text_length": len(text),
                "using_tiktoken": bool(self._tiktoken_available),
                "sample_tokens": toks[:50] if isinstance(toks, (list, tuple)) else None
            }
        except Exception as e:
            s = str(text)
            return {
                "token_count": max(1, int(len(s) / 3.3)),
                "text_length": len(s),
                "error": str(e),
                "using_tiktoken": False
            }

token_counter = TokenCounter()
