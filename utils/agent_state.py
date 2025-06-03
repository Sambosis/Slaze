from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from anthropic import Anthropic  # Assuming this is the correct import
from anthropic.types.beta import (
    BetaCacheControlEphemeralParam,
)  # Assuming this is correct


class AgentState(BaseModel):
    """
    Pydantic model to store and manage the current state of the Agent.
    """

    task: str
    context_recently_refreshed: bool = False
    refresh_count: int = 45
    refresh_increment: int = 15
    messages: List[Dict[str, Any]] = Field(
        default_factory=list
    )  # Stores the conversation history
    enable_prompt_caching: bool = True
    # betas: List[str] # Assuming COMPUTER_USE_BETA_FLAG and PROMPT_CACHING_BETA_FLAG are strings
    image_truncation_threshold: int = 1
    only_n_most_recent_images: int = 2
    step_count: int = 0
    tool_params: List[Any] = Field(default_factory=list)  # Parameters for the tools

    # Optional fields that might be part of the state but are complex objects
    # For simplicity, we can exclude them or represent them with placeholders/IDs if needed
    # display: Optional[AgentDisplayWebWithPrompt] = None # Usually not part of serializable state
    # tool_collection: Optional[ToolCollection] = None # ToolCollection itself might be complex
    # output_manager: Optional[OutputManager] = None
    # token_tracker: Optional[TokenTracker] = None
    # client: Optional[Anthropic] = None # API clients are generally not serialized

    class Config:
        # Allow arbitrary types if you decide to include complex objects like 'Anthropic' client
        # Be cautious with this, as it can make serialization/deserialization harder.
        arbitrary_types_allowed = True
