
from typing import Optional, Dict, Any, List
import logging
import time

logger = logging.getLogger(__name__)

_pydantic_v2 = False
try:
    import pydantic
    pv = getattr(pydantic, "__version__", None)
    if pv:
        major = int(pv.split(".")[0])
        _pydantic_v2 = major >= 2
except Exception:
    pv = None

if _pydantic_v2:
    from pydantic import Field, SecretStr, validator
    from pydantic_settings import BaseSettings  
else:
    from pydantic import BaseSettings, Field, SecretStr, validator  

class LLMAPMConfig(BaseSettings):

    enable_monitoring: bool = True
    sampling_rate: float = 1.0
    per_endpoint_sampling: Dict[str, float] = {}
    metrics_endpoint: str = "/metrics"

    postgresql_host: Optional[str] = None
    postgresql_port: Optional[int] = None
    postgresql_database: Optional[str] = None
    postgresql_username: Optional[str] = None
    postgresql_password: Optional[str] = None

    prometheus_port: int = 8000
    prometheus_host: str = "0.0.0.0"

    default_model: Optional[str] = Field(None, env="DEFAULT_MODEL")
    cheap_model: Optional[str] = Field(None, env="CHEAP_MODEL")
    high_quality_model: Optional[str] = Field(None, env="HIGH_QUALITY_MODEL")

    enable_auto_discovery: bool = Field(True, env="ENABLE_AUTO_DISCOVERY")
    discovery_interval_seconds: int = Field(3600, env="DISCOVERY_INTERVAL")
    fallback_pricing_strategy: str = Field("conservative", env="FALLBACK_PRICING_STRATEGY")  # conservative, aggressive, family-based
    
    default_input_price_per_1k: float = Field(0.003, env="DEFAULT_INPUT_PRICE_PER_1K")
    default_output_price_per_1k: float = Field(0.006, env="DEFAULT_OUTPUT_PRICE_PER_1K")

    token_pricing: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "gpt-35-turbo": {"input": 0.0005, "output": 0.0015},
        
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
        "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "gemini-1.5-pro": {"input": 0.0035, "output": 0.0105},
        
        "text-davinci-003": {"input": 0.02, "output": 0.02},
        "text-curie-001": {"input": 0.002, "output": 0.002},
        
    }

    model_family_patterns: Dict[str, Dict[str, float]] = {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4": {"input": 0.02, "output": 0.04},
        "gpt-3.5": {"input": 0.001, "output": 0.002},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "gemini": {"input": 0.001, "output": 0.003},
        "mini": {"input": 0.0002, "output": 0.0008},
        "turbo": {"input": 0.002, "output": 0.004},   
        "large": {"input": 0.01, "output": 0.03},     
    }

    error_rate_threshold: float = 0.05
    latency_threshold_ms: int = 3000
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    redis_url: Optional[str] = Field(None, env="REDIS_URL")
    metrics_jwt_secret: Optional[SecretStr] = Field(None, env="METRICS_JWT_SECRET")

    quota_max_requests: int = Field(10, env="QUOTA_MAX_REQUESTS")
    quota_window_seconds: int = Field(3600, env="QUOTA_WINDOW_SECONDS")
    quota_cooldown_seconds: int = Field(3600, env="QUOTA_COOLDOWN_SECONDS")

    _discovered_models: Dict[str, Dict[str, Any]] = {}
    _last_discovery: float = 0

    @validator("sampling_rate")
    def validate_sampling_rate(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("sampling_rate must be between 0.0 and 1.0")
        return v

    def get_endpoint_sampling(self, endpoint: Optional[str]) -> float:
        if not endpoint:
            return float(self.sampling_rate)
        if endpoint in self.per_endpoint_sampling:
            return float(self.per_endpoint_sampling[endpoint])
        for k, v in self.per_endpoint_sampling.items():
            if k and k in endpoint:
                return float(v)
        return float(self.sampling_rate)

    @property
    def postgresql_url(self) -> str:
        if not (
            self.postgresql_host
            and self.postgresql_port
            and self.postgresql_database
            and self.postgresql_username
            and self.postgresql_password
        ):
            return ""
        return (
            f"postgresql://{self.postgresql_username}:{self.postgresql_password}"
            f"@{self.postgresql_host}:{self.postgresql_port}/{self.postgresql_database}"
        )



config = LLMAPMConfig()
_global_config = config
