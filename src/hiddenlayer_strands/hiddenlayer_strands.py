from typing import Optional
from hiddenlayer import HiddenLayer
from hiddenlayer.types import InteractionAnalyzeResponse
from strands import Agent
from strands.event_loop.event_loop import event_loop_cycle
from strands.types._events import EventLoopStopEvent, TextStreamEvent
from strands.types.streaming import ContentBlockDelta

import logging

logger = logging.Logger(__name__)


class HiddenlayerStrands:
    def __init__(
        self,
        *,
        model: str,
        client_id: Optional[str],
        client_secret: Optional[str],
        hl_block_message: str = "Blocked by Hiddenlayer",
        hl_project_id: Optional[str] = None,
        hl_requester_id: str = "Strands Agent",
        hl_client: Optional[HiddenLayer] = None,
    ):
        """Configure HiddenLayer moderation defaults for a Strands agent instance."""

        self.model = model
        self.project_id = hl_project_id
        self.requester_id = hl_requester_id
        self.block_message = hl_block_message
        self.hl_client = hl_client or HiddenLayer(base_url="https://api.hiddenlayer.ai")

    def analyze_input(self, role: str, content: str) -> InteractionAnalyzeResponse:
        """Submit the latest user turn for HiddenLayer analysis."""

        if self.project_id:
            analysis = self.hl_client.interactions.analyze(
                metadata={"model": self.model, "requester_id": self.requester_id},
                hl_project_id=self.project_id,
                input={"messages": [{"role": role, "content": content}]},
            )
        else:
            analysis = self.hl_client.interactions.analyze(
                metadata={"model": self.model, "requester_id": self.requester_id},
                input={"messages": [{"role": role, "content": content}]},
            )

        return analysis

    def analyze_output(self, content: str) -> InteractionAnalyzeResponse:
        """Submit an assistant response for HiddenLayer analysis."""

        if self.project_id:
            analysis = self.hl_client.interactions.analyze(
                metadata={"model": self.model, "requester_id": self.requester_id},
                hl_project_id=self.project_id,
                output={"messages": [{"role": "assistant", "content": str(content)}]},
            )
        else:
            analysis = self.hl_client.interactions.analyze(
                metadata={"model": self.model, "requester_id": self.requester_id},
                output={"messages": [{"role": "assistant", "content": str(content)}]},
            )
        return analysis

    def _stream_event(self, message: str) -> TextStreamEvent:
        """Create a Strands text stream event for a moderation message."""
        return TextStreamEvent(
            delta=ContentBlockDelta(text=message),
            text=message,
        )

    def _event_loop_stop_event(self, agent: Agent, message: str) -> EventLoopStopEvent:
        """Create an event loop stop event signaling moderation intervention."""
        return EventLoopStopEvent(
            stop_reason="guardrail_intervened",
            message={"role": "assistant", "content": [{"text": message}]},
            metrics=agent.event_loop_metrics,
            request_state=None,
        )

    async def hl_event_loop_cycle(self, agent: Agent, invocation_state, structured_output_context=None):
        """Bridge the Strands event loop with HiddenLayer moderation for both input and output.

        If we can't scan inputs or outputs, log the error and continue - don't disrupt the user flow.
        """

        # Otherwise, delegate and optionally intercept / modify events
        try:
            if text := agent.messages[-1]["content"][-1].get("text"):
                analysis = self.analyze_input(role=agent.messages[-1]["role"], content=text)
                if analysis.evaluation and analysis.evaluation.action == "Block":
                    yield self._stream_event(self.block_message)
                    yield self._event_loop_stop_event(agent=agent, message=self.block_message)
                    return

                if (
                    analysis.evaluation
                    and analysis.modified_data.input.messages
                    and analysis.evaluation.action == "Redact"
                ):
                    agent.messages[-1]["content"][-1]["text"] = analysis.modified_data.input.messages[-1].content
        except Exception as e:
            logger.error(f"Unable to scan inputs with Hiddenlayer: {e}")

        events = []
        async for ev in event_loop_cycle(agent, invocation_state, structured_output_context):
            events.append(ev)

            try:
                if isinstance(ev, EventLoopStopEvent):
                    final_message = ev["stop"][1]

                    # Handled structured output case where each field in the structured output
                    # is its own output
                    if structured_output_context and structured_output_context.is_enabled:
                        outputs = final_message["content"][-1]["toolUse"]["input"]

                        for output in outputs.values():
                            analysis = self.analyze_output(output)

                            if analysis.evaluation and analysis.evaluation.action == "Block":
                                yield self._stream_event(message=self.block_message)
                                yield self._event_loop_stop_event(agent=agent, message=self.block_message)
                                return
                    else:
                        analysis = self.analyze_output(content=final_message["content"][-1]["text"])
                        if analysis.evaluation and analysis.evaluation.action == "Block":
                            yield self._stream_event(message=self.block_message)
                            yield self._event_loop_stop_event(agent=agent, message=self.block_message)
                            return

                    if (
                        analysis.evaluation
                        and analysis.modified_data.output.messages
                        and analysis.evaluation.action == "Redact"
                    ):
                        redacted_message = analysis.modified_data.output.messages[-1].content
                        yield self._stream_event(message=redacted_message)
                        yield self._event_loop_stop_event(agent=agent, message=redacted_message)
                        return
            except Exception as e:
                logger.error(f"Unable to scan inputs with Hiddenlayer: {e}")

        for event in events:
            yield event
