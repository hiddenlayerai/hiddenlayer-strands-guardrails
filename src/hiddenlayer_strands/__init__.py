from typing import Optional

import strands
from hiddenlayer import HiddenLayer

from .hiddenlayer_strands import HiddenlayerStrands


def init_hiddenlayer(
    *,
    model: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
    hl_project_id: Optional[str] = None,
    hl_requester_id: str = "Strands Agent",
    hl_client: Optional[HiddenLayer] = None,
):
    """
    Initializes and configures a HiddenLayer client for model analysis, telemetry,
    or integration purposes.

    This function creates or reuses an existing HiddenLayer client and associates it
    with a specific model and optional project context. It supports both direct client
    injection (via `hl_client`) and environment-based or credential-based initialization.

    Args:
        model (str):
            Model arn on AWS Bedrock used in the agent.
        client_id (Optional[str], default=None):
            The client ID for authenticating with HiddenLayer. If not provided, the function
            will attempt to read it from the `HIDDENLAYER_CLIENT_ID` environment variable.
        client_secret (Optional[str], default=None):
            The client secret for authenticating with HiddenLayer. If not provided, the function
            will attempt to read it from the `HIDDENLAYER_CLIENT_SECRET` environment variable.
        hl_project_id (Optional[str], default=None):
            The HiddenLayer project ID under which the model should be associated.
        hl_requester_id (str, default="Strands Agent"):
            Identifier for the requesting agent or service initiating the connection.
        hl_client (Optional[HiddenLayer], default=None):
            An existing HiddenLayer client instance to use. If provided, authentication
            and initialization parameters are ignored.

    Example:
        >>> client = init_hiddenlayer(
        ...    model="anthropic.claude-sonnet-4-20250514-v1:0",
        ... )
        # Uses HIDDENLAYER_CLIENT_ID and HIDDENLAYER_CLIENT_SECRET from the environment
    """

    hiddenlayer = HiddenlayerStrands(
        model=model,
        client_id=client_id,
        client_secret=client_secret,
        hl_project_id=hl_project_id,
        hl_requester_id=hl_requester_id,
        hl_client=hl_client,
    )
    strands.agent.agent.event_loop_cycle = hiddenlayer.hl_event_loop_cycle
