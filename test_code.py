from src.stubs.slazy_agent import Agent, AgentConfig, ToolResult

# Valid code according to contract
config = AgentConfig(
    model="gpt-4",
    max_tokens=1000,
    temperature=0.1,
    system_prompt="Test prompt"
)

agent = Agent(config=config, client=None)
result = agent.process_message("Hello world")

# This should work
tool_result = ToolResult(
    success=True,
    output="Success",
    error=None,
    metadata=None
)