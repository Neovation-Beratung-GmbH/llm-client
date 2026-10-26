"""Anthropic text client."""

import os
from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from .client import Client


class AnthropicClient(Client):
    provider = "anthropic"

    async def aclose(self) -> None:
        await self.client.close()

    def _build_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    async def _query(
        self,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        text_format: type[BaseModel] | None = None,
        max_output_tokens: int = 4096,
    ) -> str | BaseModel | None:
        request: dict[str, Any] = {
            "model": self.model.model_name,
            "system": system_prompt or "",
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": timeout,
        }
        if text_format is None:
            response = await self.client.messages.create(
                **request, temperature=temperature
            )
        else:
            response = await self.client.messages.parse(
                **request,
                output_format=text_format,
                extra_body={"temperature": temperature},
            )
        usage = response.usage
        details = getattr(usage, "output_tokens_details", None)
        self._add_response_usage(
            input_tokens=usage.input_tokens,
            cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=usage.cache_read_input_tokens or 0,
            output_tokens=usage.output_tokens,
            reasoning_tokens=getattr(details, "thinking_tokens", 0) or 0,
        )
        if text_format is not None:
            return response.parsed_output
        return "".join(block.text for block in response.content if block.type == "text")
