from src.stubs.slazy_agent import Agent, AgentConfig

# Invalid code - wrong parameter name
config = AgentConfig(
    model_name="gpt-4",  # Should be 'model'
    max_tokens=1000,
    temperature=0.1,
    system_prompt="Test prompt"
)

# Invalid code - undefined class
undefined_obj = UndefinedClass()

# Invalid code - undefined method
agent = Agent(config=config, client=None)
agent.undefined_method()  # Not defined in contract

# Invalid code - wrong function signature
result = agent.process_message("Hello", "extra_param")  # Too many args