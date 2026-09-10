from .llm_client import LLMClient
from ._usage import UsageBreakdown, CostBreakdown, Usage
from ._config import (
    PROVIDER_CATALOG,
    MODEL_CATALOG,
    get_providers,
    get_models,
    get_models_for_provider,
    get_default_model_for_provider,
)

__version__ = "1.0.0"
__author__ = "Lucas Passadore Elias"

__all__ = [
    "LLMClient",
    "UsageBreakdown",
    "CostBreakdown",
    "Usage",
    "get_providers",
    "get_models",
    "get_models_for_provider",
    "get_default_model_for_provider",
    "PROVIDER_CATALOG",
    "MODEL_CATALOG",
]