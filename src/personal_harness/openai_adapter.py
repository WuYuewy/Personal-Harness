"""OpenAI-compatible usage validation on top of PydanticAI's adapter.

Credentials and endpoints come only from user configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from personal_harness.costs import TokenUsage

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types import chat
    from pydantic_ai.models.openai import OpenAIChatModel


def create_compatible_model(
    model: str, client: AsyncOpenAI, *, uniform_input_price: bool = False
) -> OpenAIChatModel:
    """Keep usage validity when reusing the framework's OpenAI-compatible adapter."""
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIStreamedResponse
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.usage import RequestUsage

    def validate_usage(
        response: chat.ChatCompletion | chat.ChatCompletionChunk, mapped: RequestUsage
    ) -> RequestUsage:
        raw = response.usage.model_dump() if response.usage is not None else None
        if raw is not None and uniform_input_price:
            if (
                raw.get("prompt_cache_hit_tokens") is None
                and (raw.get("prompt_tokens_details") or {}).get("cached_tokens") is None
            ):
                raw["prompt_cache_hit_tokens"] = 0
        normalized = normalize_usage(raw)
        mapped.details["personal_harness_usage_valid"] = int(normalized is not None)
        if normalized is not None:
            mapped.input_tokens = normalized.input_tokens
            mapped.output_tokens = normalized.output_tokens
            mapped.cache_read_tokens = normalized.cache_read_tokens
            mapped.cache_write_tokens = normalized.cache_write_tokens
        return mapped

    class CompatibleStreamedResponse(OpenAIStreamedResponse):
        def _map_usage(self, response: chat.ChatCompletionChunk) -> RequestUsage:
            if response.usage is None:
                return super()._map_usage(response)
            # Standard SSE reports the total exactly once, in the final usage chunk.
            if self._usage_received:
                self._usage.details["personal_harness_usage_valid"] = 0
                return RequestUsage()
            return validate_usage(response, super()._map_usage(response))

    class CompatibleChatModel(OpenAIChatModel):
        @property
        def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
            return CompatibleStreamedResponse

        def _map_usage(self, response: chat.ChatCompletion) -> RequestUsage:
            return validate_usage(response, super()._map_usage(response))

    return CompatibleChatModel(model, provider=OpenAIProvider(openai_client=client))


def normalize_usage(raw: dict[str, Any] | None) -> TokenUsage | None:
    """Preserve unknown counts; support standard and DeepSeek cache-hit fields.

    Absent cache detail is unknown, because silently treating all input as full
    price could hide the difference between cache savings and SDK optimization.
    """
    if raw is None:
        return None
    prompt, completion = raw.get("prompt_tokens"), raw.get("completion_tokens")
    details = raw.get("prompt_tokens_details") or {}
    detailed_hit = details.get("cached_tokens") if isinstance(details, dict) else None
    direct_hit = raw.get("prompt_cache_hit_tokens")
    if detailed_hit is not None and direct_hit is not None and detailed_hit != direct_hit:
        return None
    hit = detailed_hit if detailed_hit is not None else direct_hit
    if any(type(count) is not int or count < 0 for count in (prompt, completion, hit)):
        return None
    assert isinstance(prompt, int) and isinstance(completion, int) and isinstance(hit, int)
    missed = raw.get("prompt_cache_miss_tokens")
    if missed is not None and (type(missed) is not int or missed + hit != prompt):
        return None
    if hit > prompt:
        return None
    # This text-only adapter has no prices for cache writes or other modalities.
    output_details = raw.get("completion_tokens_details") or {}
    if (
        raw.get("cache_creation_input_tokens")
        or (
            isinstance(details, dict)
            and (details.get("audio_tokens") or details.get("cache_write_tokens"))
        )
        or (isinstance(output_details, dict) and output_details.get("audio_tokens"))
    ):
        return None
    return TokenUsage(input_tokens=prompt, output_tokens=completion, cache_read_tokens=hit)
