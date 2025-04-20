import asyncio
from typing import Optional
from contextlib import AsyncExitStack
import logging
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rich import print as rr
from anthropic import Anthropic
from dotenv import lioad_dotenv

load_dotenv()  # load environment variables from .env

# Set up logging
logging.basicConfig(level=logging.DEBUG if '--verbose' in sys.argv else logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.debug(f"Connecting to server at {server_script_path}")
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
        )
        rr(f"Connecting to server with command: {command} {server_script_path}")
        

        logger.debug(f"Server parameters: {server_params}")

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        rr("Connected to server")
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )
        logger.debug("Session created")
        rr("Session created")
        try:
            await asyncio.wait_for(self.session.initialize(), timeout=10)
            rr("Session initialized")
        except asyncio.TimeoutError:
            logger.error("Session initialization timed out")
            rr("Session initialized")
        logger.debug("Session initialized")

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.debug(f"Available tools: {[tool.name for tool in tools]}")
        rr("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        logger.debug(f"Processing query: {query}")
        messages = [{"role": "user", "content": query}]
        rr(f"Messages: {messages}")
        response = await self.session.list_tools()
        rr(f"Response: {response}")
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]
        rr(f"Available tools: {available_tools}")
        # Initial Claude API call
        rr(messages)
        
        rr(self.anthropic.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=messages,
            tools=available_tools,
        ))
        rr(f"Claude response: {response}")
        logger.debug(f"Claude response: {response}")

        # Process response and handle tool calls
        final_text = []

        assistant_message_content = []
        for content in response.content:
            if content.type == "text":
                final_text.append(content.text)
                assistant_message_content.append(content)
            elif content.type == "tool_use":
                tool_name = content.name
                tool_args = content.input
                logger.debug(f"Calling tool {tool_name} with args {tool_args}")

                # Execute tool call
                result = await self.session.call_tool(tool_name, tool_args)
                logger.debug(f"Tool result: {result.content}")
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({"role": "assistant", "content": assistant_message_content})
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content,
                            }
                        ],
                    }
                )

                # Get next response from Claude
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-latest",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools,
                )
                logger.debug(f"Claude response after tool call: {response}")
                final_text.append(response.content[0].text)

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info("MCP Client Started!")
        rr("\nMCP Client Started!")
        rr("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                response = await self.process_query(query)
                rr("\n" + response)

            except Exception as e:
                logger.error(f"Error: {str(e)}")
                rr(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        logger.debug("Cleaning up resources")
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        rr("Usage: python client.py <path_to_server_script> [--verbose]")
        sys.exit(1)

    client = MCPClient()
    rr("Connecting to server...")
    try:
        await client.connect_to_server(sys.argv[1])
        rr("Connected to server!")
        rr("Starting chat loop...")
        await client.chat_loop()
        rr("Chat loop ended.")
    except Exception as e:
        rr(f"Error during connection or chat loop: {str(e)}")
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
