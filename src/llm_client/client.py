"""Shared interface and bookkeeping for asynchronous model clients."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Self, overload

from pydantic import BaseModel

from ._config import (
    get_default_model_for_provider,
    get_model_config,
    get_provider_config,
)
from ._usage import Usage, UsageBreakdown, get_usage


class Client(ABC):
    """Subclass to implement a provider. Instantiate a concrete client instead."""

    provider: ClassVar[str]

    def __init__(
        self,
        model_name: str | None = None,
        max_concurrency: int = 50,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        self.config = get_provider_config(self.provider)
        model_name = model_name or get_default_model_for_provider(self.provider)
        model = get_model_config(model_name)
        if model is None:
            raise ValueError(f"Unsupported model: {model_name}")
        if model.provider != self.provider:
            raise ValueError(
                f"Model {model_name} does not belong to provider {self.provider}"
            )
        self.model = model
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.usage = Usage()
        self.last_usage = Usage()
        self.client = self._build_client()

    @overload
    async def query[ResponseT: BaseModel](
        self, *, text_format: type[ResponseT], **kwargs: Any
    ) -> ResponseT: ...

    @overload
    async def query(self, *, text_format: None = None, **kwargs: Any) -> Any: ...

    async def query(
        self, *, text_format: type[BaseModel] | None = None, **kwargs: Any
    ) -> Any:
        """Query the provider and validate a requested Pydantic response model."""
        if text_format is not None and (
            not isinstance(text_format, type) or not issubclass(text_format, BaseModel)
        ):
            raise TypeError("text_format must be a Pydantic model class")
        model = kwargs.pop("model", self.model.model_name)
        if model != self.model.model_name:
            raise ValueError("Create a new client to query a different model")
        async with self._semaphore:
            response = await self._query(text_format=text_format, **kwargs)
        if text_format is None:
            return response
        if response is None:
            raise ValueError("Provider returned no structured response")
        if isinstance(response, BaseModel):
            response = response.model_dump()
        if isinstance(response, (str, bytes)):
            return text_format.model_validate_json(response)
        return text_format.model_validate(response)

    @abstractmethod
    def _build_client(self) -> Any:
        """Construct the provider's SDK client."""

    @abstractmethod
    async def _query(self, **kwargs: Any) -> Any:
        """Implement the provider's request and response handling."""

    @abstractmethod
    async def aclose(self) -> None:
        """Close the provider's SDK client."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def _add_response_usage(self, **kwargs: int) -> None:
        # Cached and reasoning tokens are already included in input/output totals.
        kwargs.setdefault(
            "total_tokens",
            kwargs.get("input_tokens", 0) + kwargs.get("output_tokens", 0),
        )
        self.last_usage = get_usage(self.model, UsageBreakdown(**kwargs))
        self.usage += self.last_usage
