"""TypeSafe AI Jev client for typed decisions."""

import os
from collections.abc import Mapping

from pydantic import BaseModel
from typesafe_sdk import AsyncTypeSafeClient, Content, Question, SystemOneResponse

from .client import Client


class TypeSafeAIClient(Client):
    provider = "typesafeai"

    def _build_client(self) -> AsyncTypeSafeClient:
        return AsyncTypeSafeClient(
            api_key=os.getenv("TYPESAFE_API_KEY"), model=self.model.model_name
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def _query(
        self,
        state: Content,
        questions: Mapping[str, Question],
        timeout: float | None = None,
        temperature: float | None = None,
        text_format: type[BaseModel] | None = None,
    ) -> SystemOneResponse:
        # Accepted for the shared interface; Jev requests do not use temperature.
        response: SystemOneResponse = await self.client.system_one(
            state=state,
            questions=questions,
            model=self.model.model_name,
            timeout=timeout,
        )
        usage = response.usage
        self._add_response_usage(
            input_tokens=(usage.input_tokens or 0) if usage else 0,
            output_tokens=(usage.output_tokens or 0) if usage else 0,
        )

        return response
        # Jev's output shape is defined by questions. The base class validates
        # the complete response envelope against text_format after usage is recorded.
