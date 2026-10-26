"""OpenAI text, structured-output, and embedding client."""

import os
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel

from ._config import EMBEDDINGS_MODEL, ModelConfig, ModelPricing
from ._usage import UsageBreakdown, get_usage
from .client import Client


class OpenAIClient(Client):
    provider = "openai"

    def _build_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    async def aclose(self) -> None:
        await self.client.close()

    async def _query(
        self,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        text_format: type[BaseModel] | None = None,
        max_output_tokens: int = 4096,
    ) -> Any:
        request: dict[str, Any] = {
            "model": self.model.model_name,
            "input": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": max_output_tokens,
            "timeout": timeout,
        }
        if not self.model.reasoning_model:
            request["temperature"] = temperature
        # The API rejects anything that is not a Pydantic model, None included.
        schema = (
            text_format
            if isinstance(text_format, type) and issubclass(text_format, BaseModel)
            else None
        )
        if schema is not None:
            request["text_format"] = schema

        response = await self.client.responses.parse(**request)

        usage = response.usage
        self._add_response_usage(
            input_tokens=usage.input_tokens if usage else 0,
            cached_tokens=(getattr(usage.input_tokens_details, "cached_tokens", 0) or 0)
            if usage
            else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=(
                getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0
            )
            if usage
            else 0,
        )

        if schema is None:
            return response.output_text

        return response.output_parsed

    async def generate_embedding(self, input: str) -> list[float]:
        async with self._semaphore:
            response = await self.client.embeddings.create(
                model=EMBEDDINGS_MODEL, input=input
            )
        # Embeddings use their own model's rate, not the configured chat model's rate.
        model = ModelConfig(
            "openai", EMBEDDINGS_MODEL, ModelPricing(0.02, 0.0, cached_input_per_1m=0.0)
        )
        self.last_usage = get_usage(
            model,
            UsageBreakdown(
                input_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )
        self.usage += self.last_usage
        return response.data[0].embedding

    async def _generate_embedding(self, input: str) -> list[float]:
        """Compatibility alias for the original embedding helper."""
        return await self.generate_embedding(input)
