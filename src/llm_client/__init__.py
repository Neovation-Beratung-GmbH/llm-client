from typing import Any, Self, overload

from pydantic import BaseModel

from ._config import (
    MODEL_CATALOG,
    PROVIDER_CATALOG,
    get_default_model_for_provider,
    get_model_config,
    get_models,
    get_models_for_provider,
    get_providers,
)
from ._usage import CostBreakdown, Usage, UsageBreakdown
from .anthropic import AnthropicClient
from .client import Client
from .google import GoogleClient
from .openai import OpenAIClient
from .typesafeai import TypeSafeAIClient

__version__ = "2.1.1"
__author__ = "Lucas Passadore Elias"

__all__ = [
    "MODEL_CATALOG",
    "PROVIDER_CATALOG",
    "AnthropicClient",
    "Client",
    "CostBreakdown",
    "GoogleClient",
    "LLMClient",
    "OpenAIClient",
    "TypeSafeAIClient",
    "Usage",
    "UsageBreakdown",
    "get_default_model_for_provider",
    "get_models",
    "get_models_for_provider",
    "get_providers",
]


client_classes: dict[str, type[Client]] = {
    "anthropic": AnthropicClient,
    "gemini": GoogleClient,
    "openai": OpenAIClient,
    "typesafeai": TypeSafeAIClient,
}


class LLMClient:
    def __init__(
        self,
        model_name: str,
        max_concurrency: int = 1,
        *,
        temperature: float | None = None,
    ) -> None:
        model_config = get_model_config(model_name)
        if model_config is None:
            raise ValueError(f"Unsupported model: {model_name}")
        provider = model_config.provider
        if provider not in client_classes:
            raise ValueError(f"Unsupported provider: {provider}")
        self.temperature = temperature
        self._client: Client = client_classes[provider](
            model_name=model_name, max_concurrency=max_concurrency
        )

    @overload
    async def query[ResponseT: BaseModel](
        self,
        *,
        text_format: type[ResponseT],
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ResponseT: ...

    @overload
    async def query(
        self,
        *,
        text_format: None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any: ...

    async def query(
        self,
        *,
        text_format: type[BaseModel] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> Any:
        effective_temperature = self.temperature if temperature is None else temperature
        if effective_temperature is not None:
            kwargs["temperature"] = effective_temperature
        return await self._client.query(text_format=text_format, **kwargs)

    @property
    def usage(self) -> Usage:
        return self._client.usage

    @property
    def last_usage(self) -> Usage:
        return self._client.last_usage

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()
