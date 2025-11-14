from typing import Optional

import strands
from hiddenlayer import HiddenLayer

from .hiddenlayer_strands import HiddenlayerStrands


def init_hiddenlayer(
    *,
    model,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    hl_project_id: Optional[str] = None,
    hl_requester_id: str = "Strands Agent",
    hl_client: Optional[HiddenLayer] = None,
):
    """Install the HiddenLayer-backed event loop for all Strands agents."""

    hiddenlayer = HiddenlayerStrands(
        model=model,
        client_id=client_id,
        client_secret=client_secret,
        hl_project_id=hl_project_id,
        hl_requester_id=hl_requester_id,
        hl_client=hl_client,
    )
    strands.agent.agent.event_loop_cycle = hiddenlayer.hl_event_loop_cycle
