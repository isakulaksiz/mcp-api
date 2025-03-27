import asyncio
import logging
import sys
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from anthropic import Anthropic
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mcp_client.log')
    ]
)
logger = logging.getLogger('mcp_client')

load_dotenv()  # load environment variables from .env
logger.info("Environment variables loaded from .env")


# ------
# Basic Client Structure
# ------
class MCPClient:
    def __init__(self):
        logger.info("Initializing MCPClient")
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()
        logger.info("Anthropic client initialized")

    # -------
    # Server Connection Management
    # -------
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        logger.info(f"Connecting to server using script: {server_script_path}")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            logger.error(f"Invalid server script type: {server_script_path}")
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        logger.info(f"Using command: {command} for server script")

        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        logger.debug(f"Server parameters: {server_params}")

        logger.info("Establishing stdio client connection")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        logger.debug("Stdio transport established")

        logger.info("Creating client session")
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        logger.info("Initializing session")
        await self.session.initialize()
        logger.info("Session initialized successfully")

        # List available tools
        logger.info("Requesting list of available tools")
        response = await self.session.list_tools()
        tools = response.tools
        tool_names = [tool.name for tool in tools]
        logger.info(f"Connected to server with tools: {tool_names}")
        print("\nConnected to server with tools:", tool_names)

    # ---------
    # Query Processing Logic
    # ---------
    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        logger.info(f"Processing query: {query}")

        messages = [
            {
                "role": "user",
                "content": query
            }
        ]
        logger.debug(f"Initial messages structure: {messages}")

        logger.info("Fetching available tools")
        response = await self.session.list_tools()
        available_tools = [{
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.inputSchema
        } for tool in response.tools]
        logger.debug(f"Available tools: {available_tools}")

        # Initial Claude API call
        logger.info("Making initial Claude API call")
        response = self.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=messages,
            tools=available_tools
        )
        logger.info("Received initial response from Claude")
        logger.debug(f"Initial response: {response}")

        # Process response and handle tool calls
        final_text = []
        logger.info("Processing Claude response")

        assistant_message_content = []
        for content in response.content:
            if content.type == 'text':
                logger.info("Processing text content from Claude")
                final_text.append(content.text)
                assistant_message_content.append(content)
                logger.debug(f"Text content: {content.text[:100]}...")
            elif content.type == 'tool_use':
                logger.info(f"Processing tool use: {content.name}")
                tool_name = content.name
                tool_args = content.input
                logger.debug(f"Tool args: {tool_args}")

                # Execute tool call
                logger.info(f"Executing tool call: {tool_name}")
                result = await self.session.call_tool(tool_name, tool_args)
                logger.info(f"Tool call completed: {tool_name}")
                logger.debug(f"Tool result: {result.content}")

                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")

                assistant_message_content.append(content)
                messages.append({
                    "role": "assistant",
                    "content": assistant_message_content
                })
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": result.content
                        }
                    ]
                })
                logger.debug(f"Updated messages after tool call: {messages}")

                # Get next response from Claude
                logger.info("Making follow-up Claude API call with tool results")
                response = self.anthropic.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=1000,
                    messages=messages,
                    tools=available_tools
                )
                logger.info("Received follow-up response from Claude")
                logger.debug(f"Follow-up response: {response}")

                final_text.append(response.content[0].text)
                logger.debug(f"Added follow-up text: {response.content[0].text[:100]}...")

        logger.info("Query processing completed")
        return "\n".join(final_text)

    # --------
    # Interactive Chat Interface
    # --------
    async def chat_loop(self):
        """Run an interactive chat loop"""
        logger.info("Starting interactive chat loop")
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()
                logger.info(f"User input: {query}")

                if query.lower() == 'quit':
                    logger.info("User requested to quit")
                    break

                logger.info("Processing user query")
                response = await self.process_query(query)
                logger.info("Query processed successfully")
                print("\n" + response)

            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")

        logger.info("Chat loop ended")

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        await self.exit_stack.aclose()
        logger.info("Cleanup completed")


# --------
# Main Entry Point
# --------
async def main():
    logger.info("Starting MCP Client application")

    if len(sys.argv) < 2:
        logger.error("Missing server script path argument")
        print("Usage: python client_logger.py <path_to_server_script>")
        sys.exit(1)

    server_script = sys.argv[1]
    logger.info(f"Server script path: {server_script}")

    client = MCPClient()
    try:
        logger.info("Connecting to server")
        await client.connect_to_server(server_script)
        logger.info("Starting chat loop")
        await client.chat_loop()
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
        print(f"Fatal error: {str(e)}")
    finally:
        logger.info("Running cleanup")
        await client.cleanup()
        logger.info("Application shutdown complete")


if __name__ == "__main__":
    logger.info("Script executed directly")
    import sys

    asyncio.run(main())