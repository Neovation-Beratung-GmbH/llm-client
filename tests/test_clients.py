import asyncio
import unittest
from decimal import Decimal
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, patch

from anthropic.types import Message
from google.genai import types
from pydantic import BaseModel, ValidationError
from typesafe_sdk import Noul, SystemOneResponse

from llm_client import (
    AnthropicClient,
    Client,
    GoogleClient,
    LLMClient,
    OpenAIClient,
    TypeSafeAIClient,
    get_default_model_for_provider,
    get_models_for_provider,
    get_providers,
)


def openai_response():
    return NS(
        output_text="hello",
        output_parsed=None,
        usage=NS(
            input_tokens=100,
            output_tokens=20,
            input_tokens_details=NS(cached_tokens=10),
            output_tokens_details=NS(reasoning_tokens=5),
        ),
    )


def make_client(cls, sdk, **kwargs):
    with patch.object(cls, "_build_client", return_value=sdk):
        return cls(**kwargs)


class ClientTests(unittest.IsolatedAsyncioTestCase):
    def test_abstract_interface(self):
        with self.assertRaises(TypeError):
            Client()
        for cls in (OpenAIClient, AnthropicClient, GoogleClient, TypeSafeAIClient):
            self.assertTrue(issubclass(cls, Client))
            client = make_client(cls, NS())
            self.assertEqual(client.model.provider, client.provider)
            self.assertIn(
                client.model.model_name, get_models_for_provider(client.provider)
            )
            self.assertEqual(
                client.model.model_name, get_default_model_for_provider(client.provider)
            )
        self.assertIn("typesafeai", get_providers())

    def test_validation_before_sdk_construction(self):
        with self.assertRaisesRegex(ValueError, "does not belong"):
            AnthropicClient(model_name="gpt-5-mini")
        with self.assertRaisesRegex(ValueError, "Unsupported model"):
            OpenAIClient(model_name="missing")
        with self.assertRaisesRegex(ValueError, "max_concurrency"):
            GoogleClient(max_concurrency=0)

    async def test_openai_text_usage_and_model_guard(self):
        sdk = NS(responses=NS(parse=AsyncMock(return_value=openai_response())))
        client = make_client(OpenAIClient, sdk)
        self.assertEqual(await client.query(prompt="hi"), "hello")
        self.assertEqual(sdk.responses.parse.call_args.kwargs["model"], "gpt-4.1-mini")
        self.assertNotIn("text_format", sdk.responses.parse.call_args.kwargs)
        self.assertEqual(client.last_usage.tokens.total_tokens, 120)
        self.assertEqual(client.usage.cost.total_cost_USD, Decimal("0.000138"))
        await client.query(prompt="hi")
        self.assertEqual(client.usage.tokens.total_tokens, 240)
        with self.assertRaisesRegex(ValueError, "different model"):
            await client.query(prompt="hi", model="gpt-5-mini")
        self.assertEqual(sdk.responses.parse.await_count, 2)

    async def test_openai_structured_reasoning(self):
        class Answer(BaseModel):
            value: str

        response = openai_response()
        response.output_parsed = Answer(value="yes")
        sdk = NS(responses=NS(parse=AsyncMock(return_value=response)))
        client = make_client(OpenAIClient, sdk, model_name="gpt-5-mini")
        self.assertEqual(
            await client.query(prompt="hi", text_format=Answer), response.output_parsed
        )
        self.assertNotIn("temperature", sdk.responses.parse.call_args.kwargs)
        self.assertIs(sdk.responses.parse.call_args.kwargs["text_format"], Answer)

    async def test_anthropic_text_and_cache_usage(self):
        response = Message.model_validate(
            {
                "id": "msg_test",
                "model": "claude-sonnet",
                "role": "assistant",
                "type": "message",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 5,
                },
            }
        )
        sdk = NS(messages=NS(create=AsyncMock(return_value=response)))
        client = make_client(AnthropicClient, sdk)
        self.assertEqual(await client.query(prompt="hi"), "hello")
        self.assertEqual(client.usage.tokens.cache_creation_input_tokens, 10)
        self.assertEqual(client.usage.tokens.cache_read_input_tokens, 5)
        self.assertEqual(client.usage.cost.total_cost_USD, Decimal("0.000639"))

    async def test_google_timeout_structured_and_thinking_usage(self):
        class Answer(BaseModel):
            value: str

        response = NS(
            text="hello",
            parsed=Answer(value="yes"),
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=100,
                candidates_token_count=20,
                thoughts_token_count=5,
                cached_content_token_count=10,
            ),
        )
        sdk = NS(aio=NS(models=NS(generate_content=AsyncMock(return_value=response))))
        client = make_client(GoogleClient, sdk)
        self.assertEqual(await client.query(prompt="hi", timeout=2.5), "hello")
        config = sdk.aio.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.http_options.timeout, 2500)
        self.assertEqual(client.last_usage.tokens.output_tokens, 25)
        self.assertEqual(client.last_usage.tokens.total_tokens, 125)
        self.assertEqual(client.last_usage.cost.total_cost_USD, Decimal("0.0000898"))
        self.assertEqual(
            await client.query(prompt="hi", text_format=Answer), response.parsed
        )

    async def test_typesafe_preserves_typed_answers_and_usage(self):
        response = SystemOneResponse.model_validate(
            {
                "model": "jev-1.14.0",
                "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )
        sdk = NS(system_one=AsyncMock(return_value=response))
        client = make_client(TypeSafeAIClient, sdk)
        questions = {"urgent": Noul(instructions="Is this urgent?")}
        result = await client.query(state="Help!", questions=questions)
        self.assertIs(result, response)
        self.assertEqual(result.nouls["urgent"].noul, 0.9)
        sdk.system_one.assert_awaited_once_with(
            state="Help!", questions=questions, model="jev-latest", timeout=None
        )
        self.assertEqual(client.usage.cost.total_cost_USD, Decimal("0.0000042"))
        self.assertEqual(client.usage.tokens.total_tokens, 120)

    async def test_anthropic_pydantic_output(self):
        class Answer(BaseModel):
            value: str

        response = NS(
            parsed_output=Answer(value="yes"),
            usage=NS(
                input_tokens=10,
                output_tokens=5,
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
            ),
        )
        sdk = NS(messages=NS(parse=AsyncMock(return_value=response)))
        client = make_client(AnthropicClient, sdk)
        answer = await client.query(prompt="hi", text_format=Answer)
        self.assertIsInstance(answer, Answer)
        self.assertEqual(answer.value, "yes")
        self.assertIs(sdk.messages.parse.call_args.kwargs["output_format"], Answer)
        self.assertEqual(client.usage.tokens.total_tokens, 15)

    async def test_typesafe_custom_response_model(self):
        class Decision(BaseModel):
            noul: float

        class Answers(BaseModel):
            urgent: Decision

        class Result(BaseModel):
            answers: Answers

        sdk = NS(
            system_one=AsyncMock(
                return_value=SystemOneResponse.model_validate(
                    {
                        "model": "jev-1.14.0",
                        "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                        "usage": {"input_tokens": 100, "output_tokens": 20},
                    }
                )
            )
        )
        client = make_client(TypeSafeAIClient, sdk)
        result = await client.query(
            state="Help!",
            questions={"urgent": Noul(instructions="Is this urgent?")},
            text_format=Result,
        )
        self.assertIsInstance(result, Result)
        self.assertEqual(result.answers.urgent.noul, 0.9)
        self.assertEqual(client.usage.tokens.input_tokens, 100)

    async def test_all_providers_validate_schema_and_missing_results(self):
        class Answer(BaseModel):
            value: int

        for cls in (OpenAIClient, AnthropicClient, GoogleClient, TypeSafeAIClient):
            with self.subTest(provider=cls.provider):
                client = make_client(cls, NS())
                client._query = AsyncMock(return_value={"value": "wrong"})
                with self.assertRaises(ValidationError):
                    await client.query(text_format=Answer)
                client._query.return_value = None
                with self.assertRaisesRegex(ValueError, "no structured response"):
                    await client.query(text_format=Answer)
                client._query.return_value = {"value": 42}
                self.assertEqual((await client.query(text_format=Answer)).value, 42)
                client._query.return_value = '{"value": 7}'
                self.assertEqual((await client.query(text_format=Answer)).value, 7)
                client._query.reset_mock()
                for invalid in (dict, {}, Answer(value=1)):
                    with self.assertRaisesRegex(TypeError, "Pydantic model class"):
                        await client.query(text_format=invalid)
                client._query.assert_not_awaited()

    async def test_embedding_uses_embedding_rate(self):
        sdk = NS(
            embeddings=NS(
                create=AsyncMock(
                    return_value=NS(
                        data=[NS(embedding=[0.1, 0.2])],
                        usage=NS(prompt_tokens=100, total_tokens=100),
                    )
                )
            )
        )
        client = make_client(OpenAIClient, sdk)
        self.assertEqual(await client.generate_embedding("hi"), [0.1, 0.2])
        self.assertEqual(client.usage.cost.total_cost_USD, Decimal("0.000002"))
        self.assertEqual(
            sdk.embeddings.create.call_args.kwargs["model"], "text-embedding-3-small"
        )

    async def test_concurrency_limit(self):
        active = maximum = 0

        async def request(**kwargs):
            nonlocal active, maximum
            active += 1
            maximum = max(active, maximum)
            await asyncio.sleep(0.01)
            active -= 1
            return openai_response()

        client = make_client(
            OpenAIClient, NS(responses=NS(parse=request)), max_concurrency=2
        )
        await asyncio.gather(*(client.query(prompt="hi") for _ in range(8)))
        self.assertEqual(maximum, 2)
        self.assertEqual(client.usage.tokens.total_tokens, 960)

    async def test_failure_does_not_record_usage_or_hold_semaphore(self):
        sdk = NS(responses=NS(parse=AsyncMock(side_effect=RuntimeError("failed"))))
        client = make_client(OpenAIClient, sdk, max_concurrency=1)
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await client.query(prompt="hi")
        self.assertEqual(client.usage.tokens.total_tokens, 0)
        sdk.responses.parse.side_effect = None
        sdk.responses.parse.return_value = openai_response()
        self.assertEqual(
            await asyncio.wait_for(client.query(prompt="hi"), timeout=1), "hello"
        )

    async def test_context_managers_close_all_providers(self):
        for cls in (OpenAIClient, AnthropicClient, GoogleClient, TypeSafeAIClient):
            close = AsyncMock()
            sdk = NS(close=close, aclose=close, aio=NS(aclose=close))
            with self.subTest(provider=cls.provider):
                with self.assertRaisesRegex(RuntimeError, "body"):
                    async with make_client(cls, sdk):
                        raise RuntimeError("body")
                close.assert_awaited_once()


class EntrypointTests(unittest.IsolatedAsyncioTestCase):
    def test_selects_each_provider(self):
        for cls in (OpenAIClient, AnthropicClient, GoogleClient, TypeSafeAIClient):
            with self.subTest(provider=cls.provider):
                model_name = get_default_model_for_provider(cls.provider)
                sdk = NS()
                with patch.object(cls, "_build_client", return_value=sdk) as build:
                    wrapper = LLMClient(model_name, max_concurrency=3)
                self.assertIsInstance(wrapper._client, cls)
                self.assertIs(wrapper._client.client, sdk)
                self.assertEqual(wrapper._client.model.model_name, model_name)
                self.assertEqual(wrapper._client._semaphore._value, 3)
                build.assert_called_once_with()

    def test_invalid_selection_does_not_construct_provider(self):
        with patch.object(OpenAIClient, "_build_client") as build:
            with self.assertRaisesRegex(ValueError, "Unsupported model"):
                LLMClient("missing")
            with self.assertRaisesRegex(ValueError, "max_concurrency"):
                LLMClient("gpt-4.1-mini", max_concurrency=0)
            build.assert_not_called()
        with (
            patch.dict("llm_client.client_classes", {}, clear=True),
            self.assertRaisesRegex(ValueError, "Unsupported provider"),
        ):
            LLMClient("gpt-4.1-mini")

    async def test_query_and_live_usage_delegation(self):
        class Answer(BaseModel):
            value: str

        response = openai_response()
        response.output_parsed = Answer(value="yes")
        sdk = NS(
            responses=NS(parse=AsyncMock(return_value=response)), close=AsyncMock()
        )
        with patch.object(OpenAIClient, "_build_client", return_value=sdk):
            wrapper = LLMClient("gpt-4.1-mini")
        self.assertEqual(wrapper.usage.tokens.total_tokens, 0)
        async with wrapper as entered:
            self.assertIs(entered, wrapper)
            result = await wrapper.query(prompt="hi", text_format=Answer)
            self.assertIsInstance(result, Answer)
            self.assertEqual(result.value, "yes")
            self.assertIs(wrapper.usage, wrapper._client.usage)
            self.assertIs(wrapper.last_usage, wrapper._client.last_usage)
            self.assertEqual(wrapper.last_usage.tokens.total_tokens, 120)
            await wrapper.query(prompt="hi")
            self.assertEqual(wrapper.usage.tokens.total_tokens, 240)
            self.assertEqual(wrapper.last_usage.tokens.total_tokens, 120)
        sdk.close.assert_awaited_once()

    async def test_cleanup_on_error_and_explicit_close(self):
        sdk = NS(close=AsyncMock())
        with patch.object(OpenAIClient, "_build_client", return_value=sdk):
            wrapper = LLMClient("gpt-4.1-mini")
        with self.assertRaisesRegex(RuntimeError, "body"):
            async with wrapper:
                raise RuntimeError("body")
        sdk.close.assert_awaited_once()
        sdk.close.reset_mock()
        await wrapper.aclose()
        sdk.close.assert_awaited_once()

    async def test_temperature_default_and_query_override(self):
        sdk = NS(responses=NS(parse=AsyncMock(return_value=openai_response())))
        with patch.object(OpenAIClient, "_build_client", return_value=sdk):
            wrapper = LLMClient("gpt-4.1-mini", temperature=0.2)
        await wrapper.query(prompt="hi")
        self.assertEqual(sdk.responses.parse.call_args.kwargs["temperature"], 0.2)
        await wrapper.query(prompt="hi", temperature=0.0)
        self.assertEqual(sdk.responses.parse.call_args.kwargs["temperature"], 0.0)
        await wrapper.query(prompt="hi")
        self.assertEqual(sdk.responses.parse.call_args.kwargs["temperature"], 0.2)

    async def test_jev_query_delegation(self):
        response = SystemOneResponse.model_validate(
            {
                "model": "jev-1.14.0",
                "answers": {"urgent": {"type": "noul", "noul": 0.9}},
                "usage": {"input_tokens": 100, "output_tokens": 20},
            }
        )
        sdk = NS(system_one=AsyncMock(return_value=response), aclose=AsyncMock())
        with patch.object(TypeSafeAIClient, "_build_client", return_value=sdk):
            wrapper = LLMClient("jev-latest", temperature=0.2)
        async with wrapper:
            questions = {"urgent": Noul(instructions="Is this urgent?")}
            result = await wrapper.query(state="Help!", questions=questions)
            self.assertIs(result, response)
            await wrapper.query(state="Help!", questions=questions, temperature=0.8)
            self.assertEqual(sdk.system_one.await_count, 2)
            sdk.system_one.assert_awaited_with(
                state="Help!", questions=questions, model="jev-latest", timeout=None
            )
        sdk.aclose.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
