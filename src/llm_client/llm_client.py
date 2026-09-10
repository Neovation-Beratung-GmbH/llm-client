import asyncio
import base64
import os
from typing import Any, Optional

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from google.genai import Client as Gemini
from google.genai import types

from pydantic import BaseModel

from ._config import *
from ._usage import *

class LLMClient: 
    model: ModelConfig
    config: ProviderConfig
    client: AsyncOpenAI | AsyncAnthropic | Gemini
    usage: Usage
    last_usage: Usage

    def __init__(
            self,  
            model_name: str = "gpt-4.1-mini", 
            max_concurrency: int = 50
    ) -> None:
        model = MODEL_CATALOG.get(model_name) 
        if model is None:
            raise ValueError(f"Unsupported model: {model_name}")
        config = PROVIDER_CATALOG.get(model.provider)
        if config is None:
            raise ValueError(f"Unsupported provider: {model.provider}")

        self.config: ProviderConfig = config
        self.model: ModelConfig = model
        self.client = self._build_client(model.provider)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.usage = Usage()
        self.last_usage = Usage()

    async def query(self, **kwargs: Any) -> Any:
        provider = self.model.provider
        # The client already knows its model; passing it again is how the two drifted apart.
        kwargs.setdefault("model", self.model.model_name)

        async with self._semaphore:
            if provider == "openai":
                response = await self._openai_query(**kwargs)
            elif provider == "anthropic":
                response = await self._anthropic_query(**kwargs)
            elif provider == "gemini":
                response = await self._gemini_query(**kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            
        return response

    async def _openai_query(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        text_format: Any = None,
        max_output_tokens: int = 4096,
    ) -> Any:
        request: dict[str, Any] = {
            "model": model,
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
        schema = text_format if isinstance(text_format, type) and issubclass(text_format, BaseModel) else None
        if schema is not None:
            request["text_format"] = schema

        response = await self.client.responses.parse(**request)

        self._add_response_usage(
            input_tokens=response.usage.input_tokens,
            cached_tokens=response.usage.input_tokens_details.cached_tokens,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=response.usage.output_tokens_details.reasoning_tokens,
        )

        if schema is None:
            return response.output_text

        try:
            return response.output_parsed
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}") from e

    async def _anthropic_query(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        timeout: Optional[float] = None,
        text_format: Any = None,
        max_output_tokens: int = 4096,
    ) -> Any:
        response = await self.client.messages.create(
            model=model,
            temperature=temperature,
            system=system_prompt or "",
            max_tokens=max_output_tokens,
            messages=[
                {"role": "user", "content": prompt},
            ],
            timeout=timeout,
        )
        # Only present when extended thinking is on.
        output_details = response.usage.output_tokens_details
        self._add_response_usage(
            input_tokens=response.usage.input_tokens,
            cache_creation_input_tokens=response.usage.cache_creation_input_tokens or 0,
            cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
            output_tokens=response.usage.output_tokens,
            reasoning_tokens=output_details.thinking_tokens if output_details else 0,
        )
        return self._extract_text(response)

    async def _gemini_query(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        timeout: Optional[float] = None,
        text_format: Any = None,
        max_output_tokens: int = 4096,
    ) -> Any:
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            system_instruction=system_prompt if system_prompt else None,
            http_options=types.HttpOptions(timeout=timeout),
        )
        contents: list[Any] = []
        contents.append(prompt)
        response = await self.client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        self._add_response_usage(
            input_tokens=response.usage_metadata.prompt_token_count or 0,
            cached_tokens=response.usage_metadata.cached_content_token_count or 0,
            output_tokens=response.usage_metadata.candidates_token_count or 0,
            reasoning_tokens=response.usage_metadata.thoughts_token_count or 0,
        )
        return self._extract_text(response)
    
    async def _generate_embedding(
        self,
        input: str,
    ) -> list[float]:

        if self.model.provider != "openai":
            raise ValueError(
                f"Provider {self.model.provider} does not support embeddings; "
                f"embeddings require the OpenAI provider ({EMBEDDINGS_MODEL})"
            )
        response = await self.client.embeddings.create(
            model=EMBEDDINGS_MODEL,
            input=input,
        )
        self._add_response_usage(response)
        return response.data[0].embedding

    def _extract_text(self, response) -> str:
        provider = self.model.provider
        if provider == "openai":
            choices = getattr(response, "choices", None) or []
            if not choices:
                return ""
            message = getattr(choices[0], "message", None)
            return (getattr(message, "content", "") or "") if message else ""

        elif provider == "anthropic":
            parts: list[str] = []
            for block in getattr(response, "content", []):
                if getattr(block, "type", None) == "text":
                    parts.append(block.text)
            return "".join(parts)

        elif provider == "gemini":
            return getattr(response, "text", "") or ""

        raise ValueError(f"Unsupported provider: {provider}")

    @staticmethod
    def _build_client(provider: str):

        if provider == "openai":
            try:
                return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        if provider == "anthropic":
            try:
                return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        if provider == "gemini":
            try:
                return Gemini(api_key=os.getenv("GEMINI_API_KEY"))
            except Exception:
                raise ValueError(f"Missing API key for provider: {provider}")

        raise ValueError(f"Unknown provider: {provider}")
    
    def _add_response_usage(self, **kwargs) -> None:
        # No provider reports a total, so derive one. Reasoning is already inside
        # output_tokens, and cached tokens inside input_tokens, so neither is added again.
        kwargs.setdefault(
            "total_tokens", kwargs.get("input_tokens", 0) + kwargs.get("output_tokens", 0)
        )
        self.last_usage = get_usage(self.model, UsageBreakdown(**kwargs))
        self.usage += self.last_usage

    
