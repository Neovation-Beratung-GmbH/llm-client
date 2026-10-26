"""Google Gemini text and structured-output client."""

import os
from typing import Any

from google import genai
from pydantic import BaseModel

from .client import Client


class GoogleClient(Client):
    provider = "gemini"

    def _build_client(self) -> genai.Client:
        return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    async def aclose(self) -> None:
        await self.client.aio.aclose()

    async def _query(
        self,
        prompt: str,
        temperature: float = 0.7,
        system_prompt: str | None = None,
        timeout: float | None = None,
        text_format: type[BaseModel] | None = None,
        max_output_tokens: int = 4096,
    ) -> Any:
        config: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_prompt,
        }
        if timeout is not None:
            # The common client interface uses seconds; Google's SDK uses milliseconds.
            config["http_options"] = genai.types.HttpOptions(
                timeout=int(timeout * 1000)
            )
        if text_format is not None:
            if not isinstance(text_format, type) or not issubclass(
                text_format, BaseModel
            ):
                raise ValueError("text_format must be a Pydantic model class")
            config.update(
                response_mime_type="application/json", response_schema=text_format
            )
        response = await self.client.aio.models.generate_content(
            model=self.model.model_name,
            contents=[prompt],
            config=genai.types.GenerateContentConfig(**config),
        )
        usage = response.usage_metadata
        self._add_response_usage(
            input_tokens=(usage.prompt_token_count or 0) if usage else 0,
            cached_tokens=(usage.cached_content_token_count or 0) if usage else 0,
            output_tokens=(
                (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            )
            if usage
            else 0,
            reasoning_tokens=(usage.thoughts_token_count or 0) if usage else 0,
        )
        return response.parsed if text_format is not None else (response.text or "")
