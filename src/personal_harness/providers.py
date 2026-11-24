"""Local model double; PydanticAI owns the actual request/tool loop."""

from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

CONTEXT_TOOL = "context_describe"


def offline_model(context: dict[str, Any]) -> FunctionModel:
    """Request a configured tool, then render its result without a remote model."""

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        current = max(
            (
                i
                for i, message in enumerate(messages)
                if isinstance(message, ModelRequest)
                and any(isinstance(part, UserPromptPart) for part in message.parts)
            ),
            default=0,
        )
        returns = [
            part
            for message in messages[current:]
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolReturnPart) and part.tool_name == CONTEXT_TOOL
        ]
        offered = {tool.name for tool in info.function_tools}
        if CONTEXT_TOOL in offered and not returns:
            return ModelResponse([ToolCallPart(CONTEXT_TOOL, {"section": "all"})])

        prompts = [
            part.content
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        ]
        task = prompts[-1] if prompts else ""
        tool_output = str(returns[-1].content) if returns else "disabled"
        output = (
            "[OFFLINE DEMO: configuration rehearsal]\n"
            f"User: {context['user']['display_name']}\n"
            f"Preferred name: {context['user']['display_name']}\n"
            f"Profile: {context['profile']}\n"
            f"Language: {context['preferences']['language']}\n"
            f"Response style: {context['preferences']['response_style']}\n"
            f"Assistant experience: {context['experience']}\n"
            f"City: {context['preferences']['city'] or '(not configured)'}\n"
            f"Task: {task}\n"
            f"Context tool: {tool_output}\n"
            "No remote model request was sent; the task has not been executed."
        )
        return ModelResponse([TextPart(output)])

    return FunctionModel(respond, model_name="offline-demo")
