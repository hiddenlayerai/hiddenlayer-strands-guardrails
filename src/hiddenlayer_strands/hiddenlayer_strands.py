import json
import logging
import os
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from ._version import __version__

import httpx
import strands.event_loop.event_loop as _strands_agent_module
from hiddenlayer import AsyncHiddenLayer
from hiddenlayer._types import RequestOptions
from pydantic import BaseModel, Field
from strands import Agent as StrandsAgent
from strands.event_loop.event_loop import _handle_model_execution
from strands.telemetry import Trace, Tracer
from strands.tools.structured_output._structured_output_context import StructuredOutputContext
from strands.types._events import (
    EventLoopStopEvent,
    TextStreamEvent,
    TypedEvent,
)
from strands.types.streaming import ContentBlockDelta

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


def _safe_json_dumps(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


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
        return _safe_json_dumps(value)
    if isinstance(value, dict):
        return _safe_json_dumps(value)
    try:
        return str(value)
    except Exception:
        return _safe_json_dumps(value)


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

    async def _submit_request_evaluation(self, body: dict, roundtrip_id: str, session_id: str | None = None) -> httpx.Response:
        return await self.hl_client.post(
            REQUEST_EVALUATIONS_PATH,
            cast_to=httpx.Response,
            body=body,
            options=self._build_options(roundtrip_id, session_id),
        )

    async def _submit_response_evaluation(self, body: dict, roundtrip_id: str, session_id: str | None = None) -> httpx.Response:
        return await self.hl_client.post(
            RESPONSE_EVALUATIONS_PATH,
            cast_to=httpx.Response,
            body=body,
            options=self._build_options(roundtrip_id, session_id),
        )

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
            resp = await self._submit_request_evaluation(request_body, roundtrip_id, session_id)
            if resp.headers.get("hl-runtime-action", "").upper() == "BLOCK":
                raise InputBlockedError(self.block_message)
        except InputBlockedError:
            raise
        except Exception:
            logger.warning("Failed to submit request evaluation to HiddenLayer", exc_info=True)

        msg_count_before = len(agent.messages)

        async for event in _handle_model_execution(
            agent, cycle_span, cycle_trace, invocation_state, tracer, structured_output_context
        ):
            yield event

        try:
            new_messages = agent.messages[msg_count_before:]
            response_body = _new_assistant_message_to_openai_response(
                new_messages, self.hiddenlayer_params.model or "unknown"
            )
            resp = await self._submit_response_evaluation(response_body, roundtrip_id, session_id)
            if resp.headers.get("hl-runtime-action", "").upper() == "BLOCK":
                raise OutputBlockedError(self.block_message)
        except OutputBlockedError:
            raise
        except Exception:
            logger.warning("Failed to submit response evaluation to HiddenLayer", exc_info=True)

    # async def hl_event_loop_cycle(self, agent: StrandsAgent, invocation_state, structured_output_context=None):
    #     """Bridge the Strands event loop with HiddenLayer moderation for both input and output."""
    #     try:
    #         if text := agent.messages[-1]["content"][-1].get("text"):
    #             response = await self._analyze_content(
    #                 [{"role": agent.messages[-1]["role"], "content": text}],
    #                 "user",
    #             )
    #             result = _parse_analysis(response, "user")

    #             if result.block:
    #                 yield self._stream_event(self.block_message)
    #                 yield self._event_loop_stop_event(agent=agent, message=self.block_message)
    #                 return

    #             if result.redact and result.redacted_content:
    #                 agent.messages[-1]["content"][-1]["text"] = result.redacted_content
    #     except Exception as e:
    #         logger.error(f"Unable to scan inputs with HiddenLayer: {e}")

    #     events = []
    #     output_result: AnalysisResult | None = None
    #     async for ev in event_loop_cycle(agent, invocation_state, structured_output_context):
    #         events.append(ev)

    #         try:
    #             if isinstance(ev, EventLoopStopEvent):
    #                 final_message = ev["stop"][1]

    #                 if structured_output_context and structured_output_context.is_enabled:
    #                     outputs = final_message["content"][-1]["toolUse"]["input"]
    #                     for output in outputs.values():
    #                         response = await self._analyze_content(
    #                             [{"role": "assistant", "content": _normalize_content(output)}],
    #                             "assistant",
    #                         )
    #                         output_result = _parse_analysis(response, "assistant")
    #                         if output_result.block:
    #                             yield self._stream_event(message=self.block_message)
    #                             yield self._event_loop_stop_event(agent=agent, message=self.block_message)
    #                             return
    #                 else:
    #                     response = await self._analyze_content(
    #                         [
    #                             {
    #                                 "role": "assistant",
    #                                 "content": _normalize_content(final_message["content"][-1]["text"]),
    #                             }
    #                         ],
    #                         "assistant",
    #                     )
    #                     output_result = _parse_analysis(response, "assistant")

    #                     if output_result.block:
    #                         yield self._stream_event(message=self.block_message)
    #                         yield self._event_loop_stop_event(agent=agent, message=self.block_message)
    #                         return

    #                 if output_result is not None and output_result.redact and output_result.redacted_content:
    #                     yield self._stream_event(message=output_result.redacted_content)
    #                     yield self._event_loop_stop_event(agent=agent, message=output_result.redacted_content)
    #                     return
    #         except Exception as e:
    #             logger.error(f"Unable to scan outputs with HiddenLayer: {e}")

    #     for event in events:
    #         yield event


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
        # _strands_agent_module.event_loop_cycle = guardrail.hl_event_loop_cycle  # type: ignore[attr-defined]
        _strands_agent_module._handle_model_execution = guardrail.handle_model_execution  # ty:ignore[unresolved-attribute]

        return StrandsAgent(**agent_kwargs)
