"""One explicitly configured upgrade after a priced, truncated response.

Framework FallbackModel owns the retry protocol. Unknown usage, billing failures,
timeouts, permissions and cancellations never trigger an upgrade.
"""

from __future__ import annotations

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from personal_harness.context import CURRENT_METRICS


def truncated(response: ModelResponse) -> bool:
    metrics = CURRENT_METRICS.get()
    if response.finish_reason == "length" and metrics is not None and metrics.model_upgrades == 0:
        metrics.model_upgrades = 1
        return True
    return False


class LimitedUpgrade(WrapperModel):
    def __init__(self, primary: Model, fallback: Model) -> None:
        super().__init__(FallbackModel(primary, fallback, fallback_on=truncated))
        self.fallback = fallback

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        metrics = CURRENT_METRICS.get()
        model = self.fallback if metrics is not None and metrics.model_upgrades else self.wrapped
        return await model.request(messages, model_settings, model_request_parameters)
