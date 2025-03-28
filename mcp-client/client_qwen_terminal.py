import asyncio
import sys
import time
import json
import logging
import datetime
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/mcp_client_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Set up file handler
file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)

# Set up console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Set up logger
logger = logging.getLogger("MCP_CLIENT")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Apply nest_asyncio to allow nested event loops
logger.info("Nest asyncio applied")


# ------
# Basic Client Structure
# ------
class MCPClient:
    def __init__(self):
        logger.info("Initializing MCPClient")
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Connect to LM Studio with Qwen model
        logger.info("Configuring OpenAI client for LM Studio connection")
        self.openai = OpenAI(
            base_url="http://localhost:1234/v1",  # Default to localhost
            api_key="lm-studio",
            timeout=180.0
        )
        self.server_script_path = None
        self.tools = []
        self.connected = False
        self.chat_history = []
        logger.debug("MCPClient initialized")

    # -------
    # Server Connection Management
    # -------
    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        logger.info("Attempting to connect to server")
        if self.connected:
            logger.info("Already connected to server")
            return [tool.name for tool in self.tools]

        self.server_script_path = server_script_path
        logger.debug(f"Server script path: {server_script_path}")

        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            logger.info(f"Invalid server script format: {server_script_path}")
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        logger.debug(f"Using command: {command} for server script")

        logger.info(f"Creating StdioServerParameters with {command} {server_script_path}")
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        logger.info("Establishing stdio client connection")
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            logger.debug("Stdio transport established")
        except Exception as e:
            logger.info(f"Failed to establish stdio transport: {str(e)}")
            raise

        logger.info("Creating client session")
        try:
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            logger.debug("Client session created")
        except Exception as e:
            logger.info(f"Failed to create client session: {str(e)}")
            raise

        logger.info("Initializing session")
        try:
            await self.session.initialize()
            logger.debug("Session initialized")
        except Exception as e:
            logger.info(f"Failed to initialize session: {str(e)}")
            raise

        # List available tools
        logger.info("Listing available tools")
        try:
            response = await self.session.list_tools()
            self.tools = response.tools
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"Found tools: {', '.join(tool_names)}")

            # Print available tools to console
            print("\nAvailable tools:")
            for tool in tool_names:
                print(f"- {tool}")
            print()

        except Exception as e:
            logger.info(f"Failed to list tools: {str(e)}")
            raise

        self.connected = True
        logger.info("Successfully connected to server")
        return [tool.name for tool in self.tools]

    # ---------
    # Query Processing Logic
    # ---------
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using Qwen and available tools"""
        logger.info(f"Processing query: {query[:50]}...")
        if not self.connected:
            logger.info("Attempted to process query but not connected to server")
            raise ValueError("Not connected to server")

        # Add user message to chat history
        logger.debug("Adding user message to chat history")
        self.chat_history.append({
            "role": "user",
            "content": query
        })

        # Format messages for the API call
        messages = self.chat_history.copy()
        logger.debug(f"Chat history size: {len(messages)} messages")

        try:
            # Format tools for OpenAI API
            logger.info("Formatting tools for OpenAI API")
            available_tools = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            } for tool in self.tools]
            logger.debug(f"Formatted {len(available_tools)} tools for API")

            # Set streaming to false to avoid connection issues
            logger.info("Calling OpenAI API for chat completion")
            try:
                logger.debug("Starting initial API call")
                print("Sending query to LM Studio...", end="", flush=True)
                response = self.openai.chat.completions.create(
                    model="lmstudio-community/qwen2.5-7b-instruct",
                    messages=messages,
                    tools=available_tools,
                    timeout=120.0,
                    stream=False  # Disable streaming to prevent disconnections
                )
                print(" Done.")
                logger.info("Initial API call completed successfully")
            except Exception as e:
                error_msg = f"LM Studio connection error: {str(e)}"
                logger.info(error_msg)
                print(f"\n⚠️ {error_msg}. Please check if LM Studio server is running at http://localhost:1234.")
                return {
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}. Please check if LM Studio server is running at http://localhost:1234."
                }

            # Process response
            logger.debug("Processing API response")
            assistant_message = response.choices[0].message
            assistant_content = assistant_message.content or ""
            logger.debug(f"Assistant response length: {len(assistant_content)} chars")

            # Create a response dict to add to chat history
            response_dict = {
                "role": "assistant",
                "content": assistant_content
            }

            # Process tool calls if any
            if hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
                logger.info(f"Assistant requested {len(assistant_message.tool_calls)} tool calls")
                tool_calls_data = []

                for idx, tool_call in enumerate(assistant_message.tool_calls):
                    tool_name = tool_call.function.name
                    logger.info(f"Processing tool call #{idx + 1}: {tool_name}")
                    print(f"\nExecuting tool: {tool_name}")

                    # Parse arguments more safely
                    logger.info(f"Parsing arguments for {tool_name}")
                    try:
                        # First try to evaluate as Python literal
                        logger.info("Attempting to parse args as Python literal")
                        tool_args = eval(tool_call.function.arguments)
                        logger.info("Successfully parsed args as Python literal")
                    except Exception as parse_error:
                        logger.info(f"Python literal parsing failed: {str(parse_error)}")
                        try:
                            # Then try to parse as JSON
                            logger.info("Attempting to parse args as JSON")
                            tool_args = json.loads(tool_call.function.arguments)
                            logger.info("Successfully parsed args as JSON")
                        except Exception as json_error:
                            # Fallback to using the string directly
                            logger.info(f"JSON parsing failed: {str(json_error)}")
                            logger.info("Using raw string as arguments")
                            tool_args = tool_call.function.arguments

                    # Log what we're about to do
                    logger.info(f"Calling tool: {tool_name}")
                    logger.info(f"Tool arguments: {tool_args}")
                    print(f"Arguments: {tool_args}")

                    try:
                        logger.info(f"Executing tool call to {tool_name}")
                        result = await self.session.call_tool(tool_name, tool_args)
                        result_content = result.content
                        logger.info(f"Tool {tool_name} executed successfully")
                        print(f"Result: {result_content}")
                    except Exception as e:
                        # Handle tool call errors
                        error_msg = f"Error calling tool: {str(e)}"
                        logger.info(f"Tool call failed: {error_msg}")
                        result_content = f"⚠️ {error_msg}"
                        print(f"Error: {error_msg}")

                    # Create a dummy Result object if needed
                    if isinstance(result_content, str):
                        logger.info("Creating dummy result object")
                        result = type('obj', (object,), {'content': result_content})

                    # Format tool call for chat history
                    logger.info("Formatting tool call for chat history")
                    tool_call_data = {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    tool_calls_data.append(tool_call_data)

                    # Add tool result to chat history
                    logger.info("Adding assistant message with tool call to chat history")
                    self.chat_history.append({
                        "role": "assistant",
                        "content": assistant_content,
                        "tool_calls": [tool_call_data]
                    })

                    logger.info("Adding tool result to chat history")
                    self.chat_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content
                    })

                    # Get next response from Qwen
                    try:
                        logger.info("Getting follow-up response after tool call")
                        print("\nGetting final response...", end="", flush=True)

                        logger.info(f"Chat history size before follow-up: {len(self.chat_history)} messages")
                        response = self.openai.chat.completions.create(
                            model="lmstudio-community/qwen2.5-7b-instruct",
                            messages=self.chat_history,
                            tools=available_tools,
                            timeout=120.0,
                            stream=False  # Disable streaming
                        )
                        print(" Done.")
                        logger.info("Follow-up response received successfully")

                        next_message = response.choices[0].message
                        assistant_content = next_message.content or ""
                        logger.info(f"Follow-up response length: {len(assistant_content)} chars")

                        # Update response dict
                        response_dict = {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    except Exception as e:
                        # Handle LM Studio connection errors for the follow-up
                        error_msg = f"LM Studio connection error after tool call: {str(e)}"
                        logger.info(error_msg)
                        print(f"\n⚠️ {error_msg}")
                        response_dict = {
                            "role": "assistant",
                            "content": f"⚠️ {error_msg}"
                        }

                # If there were tool calls, add them to the response
                if tool_calls_data:
                    logger.info("Adding tool calls data to final response")
                    response_dict["tool_calls"] = tool_calls_data

            # Add final assistant message to chat history
            logger.info("Adding final assistant message to chat history")
            self.chat_history.append(response_dict)
            logger.info("Query processing completed successfully")

            return response_dict

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.info(f"Query processing failed: {error_msg}")
            print(f"\n⚠️ {error_msg}")
            error_response = {
                "role": "assistant",
                "content": f"⚠️ {error_msg}"
            }
            self.chat_history.append(error_response)
            return error_response

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\n=== MCP Client Chat Interface ===")
        print("Type 'exit' or 'quit' to end the session")
        print("Type 'clear' to clear chat history\n")

        while True:
            try:
                # Get user input
                query = input("\nYou: ")

                # Handle exit commands
                if query.lower() in ['exit', 'quit']:
                    print("Exiting chat...")
                    break

                # Handle clear command
                if query.lower() == 'clear':
                    self.chat_history = []
                    print("Chat history cleared.")
                    continue

                # Process query
                if query:
                    response = await self.process_query(query)
                    print(f"\nAssistant: {response['content']}")
            except KeyboardInterrupt:
                print("\nReceived keyboard interrupt. Exiting...")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}", exc_info=True)
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        if self.connected:
            logger.debug("Closing async exit stack")
            await self.exit_stack.aclose()
            self.connected = False
            logger.info("Cleanup completed, disconnected from server")
        else:
            logger.debug("No active connection to clean up")


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
        print(f"Connecting to MCP server at: {server_script}")
        await client.connect_to_server(server_script)
        print("Connection successful!")
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
    asyncio.run(main())