import json
import logging
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

import httpx
import strands.event_loop.event_loop as _strands_agent_module
from hiddenlayer import AsyncHiddenLayer
from hiddenlayer._types import RequestOptions
from pydantic import BaseModel, Field
from strands import Agent as StrandsAgent
from strands.event_loop.event_loop import _handle_model_execution as _original_handle_model_execution
from strands.telemetry import Trace, Tracer
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from strands.types._events import (
    TypedEvent,
)

from ._version import __version__

REQUEST_EVALUATIONS_PATH = "/detection/v2/request-evaluations"
RESPONSE_EVALUATIONS_PATH = "/detection/v2/response-evaluations"

logger = logging.getLogger(__name__)


class HiddenLayerParams(BaseModel):
    """HiddenLayer request metadata and policy routing parameters."""

    model: str | None = None
    project_id: str | None = Field(default_factory=lambda: os.getenv("HIDDENLAYER_PROJECT_ID"))
    requester_id: str = Field(
        default_factory=lambda: os.getenv("HIDDENLAYER_REQUESTER_ID", "hiddenlayer-strands-integration")
    )


class HiddenlayerActions(str, Enum):
    BLOCK = "Block"
    REDACT = "Redact"


class InputBlockedError(Exception):
    """Raised when HiddenLayer blocks the input."""


class OutputBlockedError(Exception):
    """Raised when HiddenLayer blocks the output."""


@dataclass
class AnalysisResult:
    block: bool
    redact: bool
    redacted_content: str | None


def _normalize_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if isinstance(value, list):
        text_parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                part_type = str(part.get("type", "")).lower()
                if "text" in part_type and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif part_type == "text" and isinstance(part.get("content"), str):
                    text_parts.append(part["content"])
            elif hasattr(part, "text") and isinstance(getattr(part, "text"), str):
                text_parts.append(part.text)
        if text_parts:
            return "\n".join(text_parts)
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)

    return str(value)


def _strands_messages_to_openai(messages: list[Any]) -> list[dict[str, Any]]:
    """Convert Strands message array to OpenAI chat completions messages format."""
    result: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content_blocks = msg.get("content", [])
        if not isinstance(content_blocks, list):
            result.append({"role": role, "content": str(content_blocks)})
            continue

        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tool_use.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tool_use.get("name", ""),
                            "arguments": json.dumps(tool_use.get("input", {})),
                        },
                    }
                )
            elif "toolResult" in block:
                tool_result = block["toolResult"]
                tr_content = tool_result.get("content", [])
                tr_text = ""
                if isinstance(tr_content, list):
                    tr_text = " ".join(c["text"] for c in tr_content if isinstance(c, dict) and "text" in c)
                elif isinstance(tr_content, str):
                    tr_text = tr_content
                result.append(
                    {
                        "role": "tool",
                        "content": tr_text,
                        "tool_call_id": tool_result.get("toolUseId", ""),
                    }
                )

        if role != "tool" or text_parts or tool_calls:
            oai_msg: dict[str, Any] = {
                "role": role,
                "content": "\n".join(text_parts) if text_parts else None,
            }
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            # Only append if we have actual content (skip pure toolResult messages already emitted above)
            if text_parts or tool_calls:
                result.append(oai_msg)

    return result


def _strands_tools_to_openai(agent: Any) -> list[dict[str, Any]]:
    """Convert agent tool registry specs to OpenAI tools format."""
    try:
        specs = agent.tool_registry.get_all_tool_specs()
    except Exception:
        return []
    tools = []
    for spec in specs:
        try:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec.get("description", ""),
                        "parameters": spec["inputSchema"]["json"],
                    },
                }
            )
        except (KeyError, TypeError):
            continue
    return tools


def _new_assistant_message_to_openai_response(new_messages: list[Any], model: str) -> dict[str, Any]:
    """Build an OpenAI response payload from the first new assistant message."""
    assistant_msg: dict[str, Any] | None = None
    for msg in new_messages:
        if msg.get("role") == "assistant":
            assistant_msg = msg
            break

    if assistant_msg is None:
        return {"choices": [], "model": model}

    content_blocks = assistant_msg.get("content", [])
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    if isinstance(content_blocks, list):
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tool_use.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tool_use.get("name", ""),
                            "arguments": json.dumps(tool_use.get("input", {})),
                        },
                    }
                )

    oai_message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) if text_parts else None,
    }
    if tool_calls:
        oai_message["tool_calls"] = tool_calls

    finish_reason = "tool_calls" if tool_calls else "stop"

    return {
        "choices": [
            {
                "index": 0,
                "message": oai_message,
                "finish_reason": finish_reason,
            }
        ],
        "model": model,
    }


def _apply_request_redaction(agent: StrandsAgent, redacted_body: dict) -> None:
    """Apply redacted content from HL request-evaluation response back to agent.messages (in-place).

    The HL API returns OpenAI-format messages after redaction. We map them back to the
    Strands message format by walking both arrays simultaneously:
    - Each toolResult block in a Strands message → one "tool" OpenAI message
    - Text + toolUse blocks in a Strands message → one combined OpenAI message
    """
    redacted_messages = redacted_body.get("messages", [])

    # Apply redaction to system prompt if present
    if redacted_messages and redacted_messages[0].get("role") == "system":
        new_system = redacted_messages[0].get("content")
        if new_system and isinstance(new_system, str):
            agent.system_prompt = new_system

    redacted_non_system = [m for m in redacted_messages if m.get("role") != "system"]

    redacted_idx = 0
    for agent_msg in agent.messages:
        if redacted_idx >= len(redacted_non_system):
            break
        content_blocks = agent_msg.get("content", [])
        if not isinstance(content_blocks, list):
            continue

        # toolResult blocks each map to a separate OpenAI "tool" message
        for block in content_blocks:
            if not isinstance(block, dict) or "toolResult" not in block:
                continue
            if redacted_idx >= len(redacted_non_system):
                break
            redacted_msg = redacted_non_system[redacted_idx]
            redacted_idx += 1
            if redacted_msg.get("role") != "tool":
                continue
            new_text = redacted_msg.get("content") or ""
            tr = block["toolResult"]
            tr_content = tr.get("content", [])
            if isinstance(tr_content, list):
                updated = False
                for c in tr_content:
                    if isinstance(c, dict) and "text" in c:
                        c["text"] = new_text
                        updated = True
                        break
                if not updated:
                    tr["content"] = [{"text": new_text}]
            else:
                tr["content"] = new_text

        # text + toolUse blocks map to one combined OpenAI message
        text_blocks = [b for b in content_blocks if isinstance(b, dict) and "text" in b]
        tool_use_blocks = [b for b in content_blocks if isinstance(b, dict) and "toolUse" in b]
        if text_blocks or tool_use_blocks:
            if redacted_idx < len(redacted_non_system):
                redacted_msg = redacted_non_system[redacted_idx]
                redacted_idx += 1
                new_content = redacted_msg.get("content")
                if new_content and isinstance(new_content, str) and text_blocks:
                    text_blocks[0]["text"] = new_content


def _apply_response_redaction(new_messages: list[Any], redacted_body: dict) -> None:
    """Apply redacted content from HL response-evaluation back to the new assistant messages (in-place).

    The HL API returns an OpenAI choices payload after redaction. We update the text content
    of the first assistant message in new_messages with the redacted content.
    """
    choices = redacted_body.get("choices", [])
    if not choices:
        return
    new_content = choices[0].get("message", {}).get("content")
    if new_content is None:
        return

    for msg in new_messages:
        if msg.get("role") != "assistant":
            continue
        content_blocks = msg.get("content", [])
        if not isinstance(content_blocks, list):
            break
        for block in content_blocks:
            if isinstance(block, dict) and "text" in block:
                block["text"] = new_content
                break
        break


def _parse_analysis(response: Any, role: Literal["user", "assistant"]) -> AnalysisResult:
    try:
        action = response.evaluation.action if response.evaluation else None
        block = action == HiddenlayerActions.BLOCK
        redact = action == HiddenlayerActions.REDACT
        redacted_content = None
        if redact and response.modified_data:
            modified = response.modified_data.input if role == "user" else response.modified_data.output
            if modified and modified.messages:
                redacted_content = modified.messages[-1].content
        return AnalysisResult(block, redact, redacted_content)
    except Exception:
        logger.warning("Failed to parse HiddenLayer analysis response. Treating as non-blocking.")
        return AnalysisResult(False, False, None)


class HiddenlayerStrands:
    def __init__(
        self,
        *,
        hiddenlayer_params: HiddenLayerParams,
        hl_block_message: str = "Blocked by HiddenLayer",
        hl_client: AsyncHiddenLayer | None = None,
    ):
        self.hiddenlayer_params = hiddenlayer_params
        self.block_message = hl_block_message
        self.hl_client: AsyncHiddenLayer = hl_client or AsyncHiddenLayer()

    def _build_options(self, roundtrip_id: str, session_id: str | None = None) -> RequestOptions:
        options: RequestOptions = {
            "headers": {
                "HL-RoundTrip-Id": roundtrip_id,
                "HL-Runtime-Edge-Provider": "aws-strands-sdk",
                "HL-Runtime-Edge-Provider-Version": __version__,
            }
        }
        if self.hiddenlayer_params.project_id:
            options["headers"]["hl-project-id"] = self.hiddenlayer_params.project_id
        if session_id:
            options["headers"]["hl-runtime-session-id"] = session_id
        return options

    async def handle_model_execution(
        self,
        agent: StrandsAgent,
        cycle_span: Any,
        cycle_trace: Trace,
        invocation_state: dict[str, Any],
        tracer: Tracer,
        structured_output_context: StructuredOutputContext,
    ) -> AsyncGenerator[TypedEvent, None]:
        roundtrip_id = str(uuid4())
        session_id: str | None = getattr(getattr(agent, "_session_manager", None), "session_id", None)

        try:
            messages = _strands_messages_to_openai(agent.messages)
            if agent.system_prompt:
                messages = [{"role": "system", "content": agent.system_prompt}] + messages

            request_body = {
                "model": self.hiddenlayer_params.model or "unknown",
                "messages": messages,
                "tools": _strands_tools_to_openai(agent),
            }
            resp = await self.hl_client.post(
                REQUEST_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=request_body,
                options=self._build_options(roundtrip_id, session_id),
            )
            if resp.headers.get("hl-runtime-action", "").upper() == "BLOCK":
                raise InputBlockedError(self.block_message)
            _apply_request_redaction(agent, resp.json())
        except InputBlockedError:
            raise
        except Exception:
            logger.warning("Failed to submit request evaluation to HiddenLayer", exc_info=True)

        msg_count_before = len(agent.messages)

        async for event in _original_handle_model_execution(
            agent, cycle_span, cycle_trace, invocation_state, tracer, structured_output_context
        ):
            yield event

        try:
            new_messages = agent.messages[msg_count_before:]
            response_body = _new_assistant_message_to_openai_response(
                new_messages, self.hiddenlayer_params.model or "unknown"
            )

            resp = await self.hl_client.post(
                RESPONSE_EVALUATIONS_PATH,
                cast_to=httpx.Response,
                body=response_body,
                options=self._build_options(roundtrip_id, session_id),
            )

            if resp.headers.get("hl-runtime-action", "").upper() == "BLOCK":
                raise OutputBlockedError(self.block_message)

            _apply_response_redaction(new_messages, resp.json())
        except OutputBlockedError:
            raise
        except Exception:
            logger.warning("Failed to submit response evaluation to HiddenLayer", exc_info=True)


class Agent:
    """Drop-in replacement for strands.Agent with HiddenLayer guardrails.

    This class acts as a factory that creates a regular strands.Agent instance
    with HiddenLayer guardrails automatically configured via the event loop.
    Guardrails analyze input and output for policy violations and will block
    or redact content when violations are detected.

    Example:
        ```python
        from hiddenlayer_strands import Agent, HiddenLayerParams

        params = HiddenLayerParams(model="anthropic.claude-sonnet-4-20250514-v1:0", project_id="my-project")

        agent = Agent(
            hiddenlayer_params=params,
            system_prompt="You are a helpful assistant.",
            tools=[my_tool],
        )

        result = agent("Hello!")
        ```
    """

    def __new__(
        cls,
        hiddenlayer_params: HiddenLayerParams | None = None,
        hiddenlayer_client: AsyncHiddenLayer | None = None,
        hl_block_message: str = "Blocked by HiddenLayer",
        **agent_kwargs: Any,
    ) -> StrandsAgent:
        """Create a strands.Agent with HiddenLayer guardrails configured.

        Args:
            hiddenlayer_params: HiddenLayerParams for configuration. Defaults are used if not provided.
            hiddenlayer_client: Optional AsyncHiddenLayer client instance.
            hl_block_message: Message returned to the user when content is blocked.
            **agent_kwargs: All other arguments forwarded to strands.Agent (model, tools, system_prompt, etc.)

        Returns:
            strands.Agent: A fully configured Agent instance with HiddenLayer guardrails.
        """
        if hiddenlayer_params is None:
            model = agent_kwargs.get("model")
            model_str = str(model) if model else None
            hiddenlayer_params = HiddenLayerParams(model=model_str)
        elif hiddenlayer_params.model is None:
            model = agent_kwargs.get("model")
            if model:
                hiddenlayer_params = hiddenlayer_params.model_copy(update={"model": str(model)})

        guardrail = HiddenlayerStrands(
            hiddenlayer_params=hiddenlayer_params,
            hl_block_message=hl_block_message,
            hl_client=hiddenlayer_client,
        )
        async def _safe_handle_model_execution(*args: Any, **kwargs: Any) -> AsyncGenerator[TypedEvent, None]:
            try:
                async for event in guardrail.handle_model_execution(*args, **kwargs):
                    yield event
            except (InputBlockedError, OutputBlockedError):
                raise
            except Exception:
                logger.warning("HiddenLayer guardrail failed, falling back to original handler", exc_info=True)
                async for event in _original_handle_model_execution(*args, **kwargs):
                    yield event

        _strands_agent_module._handle_model_execution = _safe_handle_model_execution  # ty:ignore[unresolved-attribute]

        return StrandsAgent(**agent_kwargs)
