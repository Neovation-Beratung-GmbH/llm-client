# llm-client

Async Python clients for OpenAI, Anthropic, Google Gemini, and TypeSafe AI Jev.

Install dependencies with `uv sync`. Set the appropriate environment variable:
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`,
or `TYPESAFE_API_KEY`.
Environment variables must already be loaded; the package does not load `.env` files.

## Text generation

```python
import asyncio
from llm_client import LLMClient


async def main():
    async with LLMClient(model_name="gpt-4.1-mini", max_concurrency=10) as client:
        print(await client.query(prompt="Explain async Python in one sentence."))
        print(client.usage)


asyncio.run(main())
```

`LLMClient` is the public entrypoint. It selects the provider from `model_name`
and delegates queries, usage tracking, and cleanup to its provider client.
It defaults to `max_concurrency=1`; queries are always asynchronous.
Set a default sampling temperature with `LLMClient(model_name="gpt-4.1-mini", temperature=0.2)`.
Override it for an individual call with `await client.query(prompt="...", temperature=0.8)`.
When omitted, the provider's default applies. OpenAI reasoning models omit temperature;
TypeSafe AI accepts it through the shared interface but ignores it when calling Jev.

`Client` is the abstract base implemented by the provider classes. You can also
instantiate a provider directly:

| Class | Module | Default model | Query inputs |
| --- | --- | --- | --- |
| `OpenAIClient` | `llm_client.openai` | `gpt-4.1-mini` | `prompt`, optional `system_prompt` |
| `AnthropicClient` | `llm_client.anthropic` | `claude-sonnet` | `prompt`, optional `system_prompt` |
| `GoogleClient` | `llm_client.google` | `gemini-2.5-flash` | `prompt`, optional `system_prompt` |
| `TypeSafeAIClient` | `llm_client.typesafeai` | `jev-latest` | `state`, `questions` |

All classes are exported from `llm_client`. Constructors accept `model_name`
and `max_concurrency`. Direct provider clients default to their model shown above
and `max_concurrency=50`; the entrypoint requires a model name. Credentials are
read from the environment when the provider SDK client is constructed.
Models must be registered in `_config.py` and belong to the class's provider.
The existing Anthropic catalog entry is preserved; use a model ID available to
your account when configuring the catalog.

Text clients accept `temperature`, `max_output_tokens`, and `timeout` in seconds.
All text clients accept a Pydantic model class as `text_format` and return
an instance validated against that model. Anthropic uses its SDK's structured-output parser; the selected model must support structured outputs.
OpenAI provides `await client.generate_embedding(input="...")`.

## Jev decisions

```python
import asyncio
from llm_client import LLMClient
from typesafe_sdk import Choice, Noul


async def main():
    async with LLMClient(model_name="jev-latest") as client:
        response = await client.query(
            state="I was charged twice. Please fix this today.",
            questions={
                "department": Choice(
                    instructions="Which team should handle this request?",
                    criteria={
                        "billing": "Charges, payments, and subscriptions",
                        "technical": "Bugs and integration problems",
                    },
                ),
                "urgent": Noul(instructions="Is this request urgent?"),
            },
        )
        print(response.choices["department"].choice)
        print(response.nouls["urgent"].noul)


asyncio.run(main())
```

Jev returns the SDK's `SystemOneResponse`, retaining typed decisions,
probabilities, confidence where available, model ID, and usage. It accepts
`timeout` in seconds. `jev-1.14.0` is also registered for version-pinned calls.
See the [TypeSafe async SDK](https://docs.typesafe.ai/sdk/python/api/clients/async).

All clients maintain cumulative `usage` and `last_usage` for the last completed
request. Cost estimates use catalog rates; they do not represent invoices or
account-specific discounts. Added rates come from
[TypeSafe](https://docs.typesafe.ai/models) and
[Google](https://ai.google.dev/gemini-api/docs/pricing).
Use `async with` or call `await client.aclose()` to release SDK resources;
closing the entrypoint also closes its underlying provider SDK client.

## Verification

```sh
.venv/bin/python -m unittest discover -s tests -v
```

Tests patch provider `_build_client()` methods and mock SDK calls, so they need
no API credentials. This project is a library and does not install a CLI command.

```sh
uv run ruff check .
uv run ruff format --check .
```

## Pydantic response models

Pass `text_format=YourModel` to any client's `query()`. Invalid schema arguments
raise `TypeError` before an API call; invalid returned data raises Pydantic's
`ValidationError`. An absent structured result raises `ValueError` rather than
returning `None`. Omitting `text_format` keeps the normal text or Jev response.
Token usage is recorded even if validation of a completed request fails.

```python
from pydantic import BaseModel
from llm_client import AnthropicClient


class Answer(BaseModel):
    category: str
    urgent: bool


async def classify(client: AnthropicClient) -> Answer:
    return await client.query(
        prompt="Classify this ticket: I was charged twice.",
        text_format=Answer,
    )
```

For Jev, questions define the output decisions. Its `text_format` validates the
complete response envelope, rather than requesting arbitrary generated fields.
For example, a custom model may declare `answers: YourAnswersModel` and optionally
`model` and `usage`. You can pass the SDK's `SystemOneResponse` as `text_format`
to validate the standard envelope. Fields outside the chosen model follow that
model's Pydantic `extra` configuration.
