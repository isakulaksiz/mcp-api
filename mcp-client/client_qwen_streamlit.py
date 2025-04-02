import streamlit as st
import asyncio
import sys
import time
import json
import logging
import datetime
import os
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import argparse

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
import anyio

# Komut satırı argümanlarını işle
parser = argparse.ArgumentParser(description="MCP Chat")
parser.add_argument("server_script", nargs="?", type=str,
                    help="Path to the MCP server script")
args = parser.parse_args()

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
logger.info("Starting streamlit app")
logger.info(f"Command line args: {args}")


# MCPClient class
class MCPClient:
    def __init__(self, base_url="http://localhost:1234/v1"):
        logger.info("Initializing MCPClient")
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        # Connect to LM Studio with Qwen model
        logger.info(f"Configuring OpenAI client for LM Studio connection at {base_url}")
        self.openai = OpenAI(
            base_url=base_url,
            api_key="lm-studio",
            timeout=180.0
        )
        self.server_script_path = None
        self.tools = []
        self.connected = False
        self.chat_history = []
        self.last_activity_time = time.time()
        logger.debug("MCPClient initialized")

    async def check_connection(self):
        """Check if the server connection is still alive"""
        if not self.connected or not self.session:
            logger.warning("Connection check failed: Not connected")
            return False

        try:
            # A simple ping or no-op operation to check connection
            await self.session.list_tools()
            self.last_activity_time = time.time()
            logger.debug("Connection check successful")
            return True
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            return False

    async def reconnect(self):
        """Attempt to reconnect to the server"""
        if not self.server_script_path:
            logger.warning("Cannot reconnect - no server script path")
            return False

        try:
            logger.info("Attempting to reconnect to server")

            # Make sure we're fully disconnected first
            await self.cleanup()

            # Important: reinitialize the exit stack
            self.exit_stack = AsyncExitStack()

            # Add a small delay to ensure resources are fully released
            await asyncio.sleep(1)

            # Now try to connect again
            await self.connect_to_server(self.server_script_path)

            if self.connected:
                logger.info("Reconnection successful")
            return self.connected
        except Exception as e:
            logger.error(f"Reconnection failed: {str(e)}", exc_info=True)
            # Make sure we're marked as disconnected
            self.connected = False
            return False

    async def keep_alive(self):
        """Send periodic keep-alive signals to prevent connection timeout"""
        if self.connected and self.session:
            # Only send keep-alive if idle for more than 30 seconds
            if time.time() - self.last_activity_time > 30:
                try:
                    logger.debug("Sending keep-alive")
                    await self.session.list_tools()
                    self.last_activity_time = time.time()
                    logger.debug("Keep-alive successful")
                    return True
                except Exception as e:
                    logger.warning(f"Keep-alive failed: {str(e)}")
                    return False
        return None

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server"""
        logger.info(f"Attempting to connect to server: {server_script_path}")
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
        except Exception as e:
            logger.info(f"Failed to list tools: {str(e)}")
            raise

        self.connected = True
        self.last_activity_time = time.time()
        logger.info("Successfully connected to server")
        return [tool.name for tool in self.tools]

    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a query using Qwen and available tools"""
        logger.info(f"Processing query: {query[:50]}...")

        # Check connection before processing
        if not self.connected:
            logger.info("Attempted to process query but not connected to server")
            raise ValueError("Not connected to server")

        # Verify connection is still active
        if not await self.check_connection():
            logger.info("Connection lost, attempting to reconnect...")
            reconnect_success = await self.reconnect()
            if not reconnect_success:
                return {
                    "role": "assistant",
                    "content": "⚠️ Connection to the server was lost and could not be restored. Please reconnect using the sidebar controls."
                }

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
                status_placeholder = st.empty()
                status_placeholder.info("Sending query to LM Studio...")

                response = self.openai.chat.completions.create(
                    model="lmstudio-community/qwen2.5-7b-instruct",
                    messages=messages,
                    tools=available_tools,
                    timeout=120.0,
                    stream=False  # Disable streaming to prevent disconnections
                )
                status_placeholder.empty()
                logger.info("Initial API call completed successfully")
            except Exception as e:
                error_msg = f"LM Studio connection error: {str(e)}"
                logger.info(error_msg)
                st.error(f"⚠️ {error_msg}. Please check if LM Studio server is running at http://localhost:1234.")
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

                    # Verify connection before tool call
                    if not await self.check_connection():
                        logger.warning("Connection lost before tool call, attempting to reconnect...")
                        reconnect_success = await self.reconnect()
                        if not reconnect_success:
                            return {
                                "role": "assistant",
                                "content": "⚠️ Connection to the server was lost during processing. Please reconnect using the sidebar controls."
                            }

                    # Show tool execution in UI with tool name
                    tool_status = st.empty()
                    tool_status.info(f"🛠️ Executing tool: **{tool_name}**")

                    # Parse arguments more safely - matching the terminal version exactly
                    logger.info(f"Parsing arguments for {tool_name}")
                    try:
                        # Log the raw arguments for debugging
                        logger.debug(f"Raw arguments: {tool_call.function.arguments}")

                        # Birebir terminal versiyonunu taklit et
                        try:
                            # First try to parse as JSON
                            logger.info("Attempting to parse args as JSON")
                            tool_args = json.loads(tool_call.function.arguments)
                            logger.info("Successfully parsed args as JSON")
                        except Exception as json_error:
                            logger.info(f"JSON parsing failed: {str(json_error)}")
                            try:
                                # Then try to evaluate as Python literal
                                logger.info("Attempting to parse args as Python literal")
                                tool_args = eval(tool_call.function.arguments)
                                logger.info("Successfully parsed args as Python literal")
                            except Exception as parse_error:
                                # Fallback to using the string directly
                                logger.info(f"Python literal parsing failed: {str(parse_error)}")
                                logger.info("Using raw string as arguments")
                                tool_args = tool_call.function.arguments
                    except Exception as e:
                        logger.error(f"Argument parsing failed completely: {str(e)}", exc_info=True)
                        tool_args = tool_call.function.arguments

                    # Log what we're about to do
                    logger.info(f"Calling tool: {tool_name}")
                    logger.info(f"Tool arguments type: {type(tool_args)}")
                    logger.info(f"Tool arguments: {tool_args}")
                    st.code(f"Arguments: {tool_args}", language="json")

                    try:
                        logger.info(f"Executing tool call to {tool_name}")
                        # Debug info - print more details
                        logger.debug(f"Tool call details - Name: {tool_name}, Args type: {type(tool_args)}")
                        if isinstance(tool_args, dict):
                            logger.debug(f"Tool args keys: {list(tool_args.keys())}")

                        # Wrap the call in a try-except for ClosedResourceError specifically
                        try:
                            result = await self.session.call_tool(tool_name, tool_args)
                            self.last_activity_time = time.time()  # Update activity time
                            result_content = result.content
                            logger.info(f"Tool {tool_name} executed successfully")
                            tool_status.success(f"✅ Tool **{tool_name}** executed successfully!")
                            st.code(f"Result: {result_content}", language="json")
                        except anyio.ClosedResourceError:
                            logger.error("Connection closed during tool call, attempting to reconnect")
                            reconnect_success = await self.reconnect()
                            if reconnect_success:
                                # Try the call again after reconnection
                                result = await self.session.call_tool(tool_name, tool_args)
                                self.last_activity_time = time.time()
                                result_content = result.content
                                logger.info(f"Tool {tool_name} executed successfully after reconnection")
                                tool_status.success(f"✅ Tool **{tool_name}** executed successfully after reconnection!")
                                st.code(f"Result: {result_content}", language="json")
                            else:
                                raise Exception("Failed to reconnect to server")
                    except Exception as e:
                        # Handle tool call errors with more detailed logging
                        error_msg = f"Error calling tool: {str(e)}"
                        logger.error(f"Tool call failed: {error_msg}", exc_info=True)  # Full traceback
                        result_content = f"⚠️ {error_msg}"
                        tool_status.error(f"❌ Tool **{tool_name}** failed: {error_msg}")

                        # Create a dummy Result object
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

                    # Check connection again before getting follow-up response
                    if not await self.check_connection():
                        logger.warning("Connection lost after tool call, attempting to reconnect...")
                        reconnect_success = await self.reconnect()
                        if not reconnect_success:
                            return {
                                "role": "assistant",
                                "content": "⚠️ Connection to the server was lost after tool execution. Please reconnect using the sidebar controls."
                            }

                    # Get next response from Qwen
                    try:
                        logger.info("Getting follow-up response after tool call")
                        status_placeholder = st.empty()
                        status_placeholder.info("Getting final response...")

                        logger.info(f"Chat history size before follow-up: {len(self.chat_history)} messages")
                        response = self.openai.chat.completions.create(
                            model="lmstudio-community/qwen2.5-7b-instruct",
                            messages=self.chat_history,
                            tools=available_tools,
                            timeout=120.0,
                            stream=False  # Disable streaming
                        )
                        status_placeholder.empty()
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
                        st.error(f"⚠️ {error_msg}")
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
            self.last_activity_time = time.time()  # Update activity time

            return response_dict

        except anyio.ClosedResourceError:
            logger.error("Connection to server closed unexpectedly")
            self.connected = False
            error_response = {
                "role": "assistant",
                "content": "⚠️ Connection to the server was lost. Please reconnect using the sidebar controls."
            }
            self.chat_history.append(error_response)
            return error_response
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.info(f"Query processing failed: {error_msg}")
            st.error(f"⚠️ {error_msg}")
            error_response = {
                "role": "assistant",
                "content": f"⚠️ {error_msg}"
            }
            self.chat_history.append(error_response)
            return error_response

    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up resources")
        if self.connected:
            logger.debug("Closing async exit stack")
            try:
                # Set connection state to False before closing to prevent reconnection attempts
                self.connected = False

                # More careful cleanup with explicit cancellation of tasks
                # await self.exit_stack.aclose()

                # Add a small delay to allow cleanup to complete
                await asyncio.sleep(0.5)

                logger.info("Cleanup completed, disconnected from server")
            except Exception as e:
                logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
                # Make sure connected is False even if cleanup fails
                self.connected = False
        else:
            logger.debug("No active connection to clean up")


# Function to run async code in Streamlit
async def connect_to_server(client, server_path):
    try:
        # If already connected, clean up first to prevent resource leaks
        if client.connected:
            logger.info("Cleaning up existing connection before connecting")
            await client.cleanup()
            # Add a small delay to ensure cleanup completes
            await asyncio.sleep(1)

        logger.info(f"Attempting to connect to server at: {server_path}")
        #st.info(f"Connecting to server at: {server_path}")

        # First check if the file exists
        if not os.path.exists(server_path):
            error_msg = f"Server script not found: {server_path}"
            logger.error(error_msg)
            st.error(error_msg)
            return False

        tools = await client.connect_to_server(server_path)
        st.session_state['connected'] = True
        st.session_state['tools'] = tools

        logger.info(f"Connected successfully, found {len(tools)} tools")
        return True
    except Exception as e:
        error_msg = f"Connection error: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg, exc_info=True)
        # Ensure client is marked as disconnected if connection fails
        client.connected = False
        return False


async def process_message(client, user_input):
    with st.spinner("Processing..."):
        response = await client.process_query(user_input)
    return response


async def keep_alive_task(client):
    """Background task to keep the connection alive"""
    while client.connected:
        await client.keep_alive()
        await asyncio.sleep(15)  # Check every 15 seconds


# Main Streamlit app
def main():
    st.set_page_config(
        page_title="MCP Client Interface",
        page_icon="🤖",
        layout="wide",
    )

    st.title("MCP Chat")

    # İlk olarak session_state değişkenlerini başlat
    for key in ['lm_studio_base_url', 'connected', 'tools', 'messages', 'auto_connected']:
        if key not in st.session_state:
            st.session_state[key] = {
                'lm_studio_base_url': "http://localhost:1234/v1",
                'connected': False,
                'tools': [],
                'messages': [],
                'auto_connected': False
            }[key]

    # Client'ı başlat - değişkenler başlatıldıktan sonra
    if 'client' not in st.session_state:
        st.session_state['client'] = MCPClient(base_url=st.session_state.lm_studio_base_url)

    # Eğer command line argümanı verilmişse ve henüz bağlanmadıysak, otomatik bağlan
    if args.server_script and not st.session_state.auto_connected:
        #st.info(f"Auto-connecting to server from command line argument: {args.server_script}")
        with st.spinner("Connecting to server from command line argument..."):
            result = asyncio.run(connect_to_server(st.session_state.client, args.server_script))
            if result:
                st.success(f"Auto-connected to server: {args.server_script}")
                st.session_state.auto_connected = True

    # Sidebar for configuration
    with st.sidebar:
        st.header("Server Configuration")
        default_path = args.server_script if args.server_script else "/Users/isakulaksiz/Desktop/mcp-server/weather/tools.py"
        server_path = st.text_input(
            "Server Script Path",
            value=default_path,
            help="Path to your MCP server script"
        )

        # LM Studio configuration
        st.subheader("LM Studio Configuration")
        lm_studio_url = st.text_input(
            "LM Studio Base URL",
            value=st.session_state.lm_studio_base_url,
            help="URL of your LM Studio server"
        )

        # Update LM Studio URL if changed
        if lm_studio_url != st.session_state.lm_studio_base_url:
            if st.session_state.connected:
                st.warning("LM Studio URL changed. Please reconnect to apply changes.")
                # First clean up existing client
                asyncio.run(st.session_state.client.cleanup())

            # Update the URL in session state
            st.session_state.lm_studio_base_url = lm_studio_url
            # Create new client with new URL
            st.session_state.client = MCPClient(base_url=lm_studio_url)
            st.session_state.connected = False

        if not st.session_state.connected:
            if st.button("Connect to Server"):
                with st.spinner("Connecting to server..."):
                    result = asyncio.run(connect_to_server(st.session_state.client, server_path))
                if result:
                    st.success("Connected successfully!")
        else:
            st.success("Connected to server")
            st.subheader("Available Tools")
            for tool in st.session_state.tools:
                st.markdown(f"- **{tool}**")

            if st.button("Disconnect"):
                try:
                    asyncio.run(st.session_state.client.cleanup())
                except Exception as e:
                    st.warning(f"Warning during disconnect: {str(e)}")
                    logger.warning(f"Warning during disconnect: {str(e)}")
                finally:
                    # Always update state even if cleanup fails
                    st.session_state.connected = False
                    st.session_state.tools = []
                    st.rerun()

            if st.button("Check Connection"):
                connection_status = asyncio.run(st.session_state.client.check_connection())
                if connection_status:
                    st.success("Connection is active!")
                else:
                    st.error("Connection has been lost. Try reconnecting.")

        if st.button("Clear Chat History"):
            st.session_state.client.chat_history = []
            st.session_state.messages = []
            st.rerun()

        # Debug bilgisi göster
        with st.expander("Debug Info"):
            st.write(f"Command Line Args: {args}")
            st.write(f"Connected: {st.session_state.connected}")
            st.write(f"Auto Connected: {st.session_state.auto_connected}")
            st.write(f"LM Studio URL: {st.session_state.lm_studio_base_url}")
            st.write(f"Tools: {st.session_state.tools}")
            if st.session_state.connected:
                last_activity = time.time() - st.session_state.client.last_activity_time
                st.write(f"Last activity: {last_activity:.1f} seconds ago")

    # Chat interface
    if not st.session_state.connected:
        if args.server_script:
            st.info(f"Attempting to connect to server from command line: {args.server_script}")
        else:
            st.info("Please connect to an MCP server using the sidebar options.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            elif message["role"] == "assistant":
                with st.chat_message("assistant"):
                    st.write(message["content"])
            elif message["role"] == "tool":
                # Tool adını göster
                tool_name = message.get("tool_name", "Unknown Tool")

                with st.chat_message("system"):
                    st.markdown(f"**🛠️ Tool Call: `{tool_name}`**")
                    st.code(message["content"], language="json")

        # Chat input
        user_input = st.chat_input("Type your message here...")
        if user_input:
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": user_input})

            # Display user message
            with st.chat_message("user"):
                st.write(user_input)

            # Process the message and get response
            response = asyncio.run(process_message(st.session_state.client, user_input))

            # Display assistant response
            with st.chat_message("assistant"):
                st.write(response["content"])

            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["content"]})

            # If there were tool calls, add them to the displayed history
            if "tool_calls" in response:
                try:
                    # Find tool messages in the chat history
                    for idx, msg in enumerate(st.session_state.client.chat_history):
                        if msg["role"] == "tool":
                            # Check if this tool message is already in our display history
                            if not any(m.get("tool_call_id", None) == msg.get("tool_call_id", None)
                                       for m in st.session_state.messages if m["role"] == "tool"):
                                # Tool adını bul
                                tool_name = "Unknown Tool"
                                tool_call_id = msg.get("tool_call_id", "")
                                for hist_msg in st.session_state.client.chat_history:
                                    if hist_msg.get("role") == "assistant" and "tool_calls" in hist_msg:
                                        for tool_call in hist_msg.get("tool_calls", []):
                                            if tool_call.get("id") == tool_call_id:
                                                tool_name = tool_call.get("function", {}).get("name", "Unknown Tool")
                                                break

                                st.session_state.messages.append({
                                    "role": "tool",
                                    "content": msg["content"],
                                    "tool_call_id": tool_call_id,
                                    "tool_name": tool_name
                                })
                except Exception as e:
                    logger.error(f"Error processing tool messages: {str(e)}", exc_info=True)

            # Force a rerun to update the UI
            st.rerun()


if __name__ == "__main__":
    main()