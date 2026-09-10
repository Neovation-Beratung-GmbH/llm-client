from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ProviderConfig", "ModelConfig", "ModelPricing", "PROVIDER_CATALOG", "MODEL_CATALOG", "EMBEDDINGS_MODEL", "get_providers", "get_models", "get_models_for_provider", "get_default_model_for_provider"]

@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    default_model: str

@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_name: str
    pricing: ModelPricing
    # Reasoning models reject `temperature`, and their reasoning counts toward the output cap.
    reasoning_model: bool = False
    notes: str | None = None

@dataclass(frozen=True)
class ModelPricing:
    input_per_1m: float
    output_per_1m: float
    cached_input_per_1m: float | None = None
    training_per_1m: float | None = None
    cache_creation_input_per_1m: float | None = None
    cache_read_input_per_1m: float | None = None

# The only supported embedding model; embeddings always go through OpenAI.
EMBEDDINGS_MODEL = "text-embedding-3-small"

PROVIDER_CATALOG: dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
                provider="openai",
                default_model="gpt-4.1-mini",
    ),
    "anthropic": ProviderConfig(
                provider="anthropic",
                default_model="claude-sonnet",
    ),
    "gemini": ProviderConfig(
                provider="gemini",
                default_model="gemini-1.5-pro",
    )
}

MODEL_CATALOG: dict[str, ModelConfig] = {
    "gpt-4.1-mini": ModelConfig(
                        provider="openai",
                        model_name="gpt-4.1-mini",
                        pricing=ModelPricing(
                            input_per_1m=0.80, 
                            cached_input_per_1m=0.20, 
                            output_per_1m=3.20, 
                            training_per_1m=5.00
                        ),
                        reasoning_model=False),
    "gpt-5-mini": ModelConfig(
                        provider="openai",
                        model_name="gpt-5-mini",
                        pricing=ModelPricing(
                            input_per_1m=0.25, 
                            cached_input_per_1m=0.025, 
                            output_per_1m=2.0, 
                            training_per_1m=None
                        ),
                        reasoning_model=True),
    "claude-sonnet": ModelConfig(
                        provider="anthropic",
                        model_name="claude-sonnet",
                        pricing=ModelPricing(
                            input_per_1m=3.00, 
                            output_per_1m=15.00, 
                            cache_creation_input_per_1m=3.75, 
                            cache_read_input_per_1m=0.30
                        ),
                        reasoning_model=False),
}

def get_providers() -> list[str]:
    return list(PROVIDER_CATALOG.keys())

def get_models() -> list[str]:
    return list(MODEL_CATALOG.keys())

def get_models_for_provider(provider: str) -> list[str]:
    return [model_name for model_name, cfg in MODEL_CATALOG.items() if cfg.provider == provider]

def get_default_model_for_provider(provider: str) -> str | None:
    config = PROVIDER_CATALOG.get(provider)
    if config is None:
        return None
    return config.default_model