#!/usr/bin/env python3
"""
Example usage of the data contract system.
This demonstrates how to use the generated stubs and validate code.
"""

# Import from generated stubs (this ensures contract compliance)
from src.stubs.slazy_agent import (
    Agent,
    AgentConfig,
    AgentMessage,
    ToolResult,
    ToolCall,
    DEFAULT_MODEL,
    MAX_TOKENS,
    DEFAULT_TEMPERATURE
)

def create_agent_example():
    """Example of creating an agent using contract-defined types."""
    
    # Create agent configuration using contract constants
    config = AgentConfig(
        model=DEFAULT_MODEL,
        max_tokens=MAX_TOKENS,
        temperature=DEFAULT_TEMPERATURE,
        system_prompt="You are a helpful assistant.",
        tools_enabled=["bash", "edit"]
    )
    
    # Create agent instance
    agent = Agent(
        config=config,
        client=None  # Would be actual client in real usage
    )
    
    # Process a message
    response = agent.process_message("Hello, how can I help you?")
    
    # Create a tool call
    tool_call = ToolCall(
        id="call_123",
        name="bash",
        arguments={"command": "ls -la"}
    )
    
    # Execute tool
    result = agent.execute_tool(tool_call)
    
    return agent, response, result

def demonstrate_validation():
    """Demonstrate how contract validation works."""
    from contract_agent import ContractAgent
    
    # Create contract agent for validation
    contract_agent = ContractAgent()
    
    # Example of valid code
    valid_code = '''
from src.stubs.slazy_agent import AgentConfig, DEFAULT_MODEL

config = AgentConfig(
    model=DEFAULT_MODEL,
    max_tokens=1000,
    temperature=0.1,
    system_prompt="Test"
)
'''
    
    # Example of invalid code
    invalid_code = '''
from src.stubs.slazy_agent import AgentConfig

config = AgentConfig(
    model_name="gpt-4",  # Wrong parameter name!
    max_tokens=1000,
    temperature=0.1,
    system_prompt="Test"
)
'''
    
    print("=== VALIDATION DEMO ===")
    
    print("\n1. Validating VALID code:")
    is_valid, violations = contract_agent.validate_generated_code(valid_code)
    if is_valid:
        print("✅ Code is valid!")
    else:
        print("❌ Code has violations:")
        for violation in violations:
            print(f"   • {violation}")
    
    print("\n2. Validating INVALID code:")
    is_valid, violations = contract_agent.validate_generated_code(invalid_code)
    if is_valid:
        print("✅ Code is valid!")
    else:
        print("❌ Code has violations:")
        for violation in violations:
            print(f"   • {violation}")
        
        print("\n💡 Suggested corrections:")
        suggestions = contract_agent.suggest_corrections(violations)
        for suggestion in suggestions:
            print(f"   • {suggestion}")

if __name__ == "__main__":
    print("🚀 Data Contract System Example")
    print("=" * 40)
    
    # Run the agent example
    print("\n1. Creating agent with contract-compliant code...")
    try:
        agent, response, result = create_agent_example()
        print("✅ Agent created successfully!")
        print(f"   Model: {agent.config.model}")
        print(f"   Max tokens: {agent.config.max_tokens}")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Run validation demo
    print("\n2. Running validation demo...")
    try:
        demonstrate_validation()
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Example completed!")
    print("\nTo use this system:")
    print("1. Define your API in contract.yml")
    print("2. Run 'make generate' to create stubs")
    print("3. Import from generated stubs in your code")
    print("4. Run 'make validate' to check compliance")
    print("5. Use contract_agent.py for LLM integration")