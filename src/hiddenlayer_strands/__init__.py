from hiddenlayer import HiddenLayer
from strands.types._events import ModelMessageEvent, EventLoopStopEvent
import strands

from strands import Agent
from strands.event_loop.event_loop import event_loop_cycle

from typing import Optional


class HiddenlayerStrands:
    def __init__(
        self,
        *,
        model: str,
        client_id: Optional[str],
        client_secret: Optional[str],
        hl_project_id: Optional[str] = None,
        hl_requester_id: str = "Strands Agent",
        hl_client: Optional[HiddenLayer] = None,
    ):
        self.model = model
        self.project_id = hl_project_id
        self.requester_id = hl_requester_id
        self.hl_client = hl_client or HiddenLayer(base_url="https://api.stage.hiddenlayer.ai")

    async def new_event_loop_cycle(self, agent: Agent, invocation_state, structured_output_context=None):
        # Otherwise, delegate and optionally intercept / modify events
        print(agent.messages[-1])
        if text := agent.messages[-1]["content"][-1].get("text"):
            if self.project_id:
                analysis = self.hl_client.interactions.analyze(
                    metadata={"model": self.model, "requester_id": self.requester_id},
                    hl_project_id=self.project_id,
                    input={"messages": [{"role": agent.messages[-1]["role"], "content": text}]},
                )
            else:
                analysis = self.hl_client.interactions.analyze(
                    metadata={"model": self.model, "requester_id": self.requester_id},
                    input={"messages": [{"role": agent.messages[-1]["role"], "content": text}]},
                )

            # TODO: Need to stream a SSE with key 'Data' as well
            if analysis.evaluation and analysis.evaluation.action == "Block":
                yield EventLoopStopEvent(
                    stop_reason="end_turn",
                    message={"role": "assistant", "content": [{"text": "Blocked by Hiddenlayer"}]},
                    metrics=agent.event_loop_metrics,
                    request_state=None,
                )
                return

        events = []
        async for ev in event_loop_cycle(agent, invocation_state, structured_output_context):
            events.append(ev)
            if isinstance(ev, EventLoopStopEvent):
                final_message = ev["stop"][1]
                if self.project_id:
                    analysis = self.hl_client.interactions.analyze(
                        metadata={"model": "strands-sdk", "requester_id": "test-app"},
                        hl_project_id=self.project_id,
                        output={"messages": [{"role": "assistant", "content": final_message["content"][-1]["text"]}]},
                    )
                else:
                    analysis = self.hl_client.interactions.analyze(
                        metadata={"model": "strands-sdk", "requester_id": "test-app"},
                        output={"messages": [{"role": "assistant", "content": final_message["content"][-1]["text"]}]},
                    )
                # TODO: Need to stream a SSE with key 'Data' as well
                if analysis.evaluation and analysis.evaluation.action == "Block":
                    yield EventLoopStopEvent(
                        stop_reason="end_turn",
                        message={"role": "assistant", "content": [{"text": "Blocked by Hiddenlayer"}]},
                        metrics=agent.event_loop_metrics,
                        request_state=None,
                    )
                    return

                for event in events:
                    yield event


def init_hiddenlayer(
    *,
    model,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    hl_project_id: Optional[str] = None,
    hl_requester_id: str = "Strands Agent",
    hl_client: Optional[HiddenLayer] = None,
):
    hiddenlayer = HiddenlayerStrands(
        model=model,
        client_id=client_id,
        client_secret=client_secret,
        hl_project_id=hl_project_id,
        hl_requester_id=hl_requester_id,
        hl_client=hl_client,
    )
    strands.agent.agent.event_loop_cycle = hiddenlayer.new_event_loop_cycle
