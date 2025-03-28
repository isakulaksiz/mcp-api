import streamlit as st
import subprocess
import os
import json
import time
import tempfile
import logging
import datetime
from pathlib import Path
import threading
import queue
import sys

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"{log_dir}/mcp_streamlit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
logger = logging.getLogger("MCP_STREAMLIT")
logger.setLevel(logging.DEBUG)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Starting MCP Streamlit Interface")


# Create a lightweight wrapper for the terminal client
class MCPClientWrapper:
    def __init__(self):
        logger.info("Initializing MCPClientWrapper")
        self.process = None
        self.connected = False
        self.server_script = None
        self.tools = []
        self.python_path = sys.executable
        self.terminal_client_path = None  # Will be set later
        self.output_queue = queue.Queue()
        self.reader_thread = None
        self.command_queue = queue.Queue()
        self.writer_thread = None

    def connect(self, terminal_client_path, server_script_path):
        """Connect to the MCP server using the terminal client in a subprocess"""
        if self.process and self.process.poll() is None:
            logger.info("Stopping existing client process")
            self.stop()

        logger.info(f"Starting client process with {terminal_client_path} {server_script_path}")
        self.terminal_client_path = terminal_client_path
        self.server_script = server_script_path

        try:
            # Start the process
            self.process = subprocess.Popen(
                [self.python_path, terminal_client_path, server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )

            # Start the reader thread
            self.reader_thread = threading.Thread(
                target=self._read_output,
                daemon=True
            )
            self.reader_thread.start()

            # Start the writer thread
            self.writer_thread = threading.Thread(
                target=self._write_input,
                daemon=True
            )
            self.writer_thread.start()

            # Wait for connection to complete and tools to be listed
            logger.info("Waiting for connection to complete...")
            tools_found = False
            timeout = 30  # 30 seconds timeout
            start_time = time.time()

            while not tools_found and time.time() - start_time < timeout:
                try:
                    output = self.output_queue.get(timeout=1)
                    if "Available tools:" in output:
                        self.connected = True
                        # Extract tools
                        tools_section = output.split("Available tools:")[1].strip()
                        tool_lines = tools_section.split("\n")
                        for line in tool_lines:
                            if line.startswith("- "):
                                self.tools.append(line[2:].strip())
                        tools_found = True
                        logger.info(f"Found tools: {', '.join(self.tools)}")
                except queue.Empty:
                    pass

            if not tools_found:
                raise TimeoutError("Timed out waiting for tools to be listed")

            logger.info("Connection successful")
            return self.tools

        except Exception as e:
            logger.error(f"Failed to start client process: {str(e)}", exc_info=True)
            if self.process:
                self.process.terminate()
                self.process = None
            raise

    def send_query(self, query):
        """Send a query to the client process"""
        if not self.process or self.process.poll() is not None:
            logger.error("No active client process")
            raise RuntimeError("Not connected to any server")

        logger.info(f"Sending query: {query[:50]}...")

        # Clear the output queue to start fresh
        while not self.output_queue.empty():
            self.output_queue.get_nowait()

        # Send the query
        self.command_queue.put(query)

        # Collect all output until we get a response
        responses = []
        tool_calls = []
        tool_results = []
        final_response = None
        response_received = False

        timeout = 120  # 2 minute timeout
        start_time = time.time()

        while not response_received and time.time() - start_time < timeout:
            try:
                output = self.output_queue.get(timeout=1)
                responses.append(output)

                # Check for tool execution
                if "Executing tool:" in output:
                    tool_name = output.split("Executing tool:")[1].strip()
                    tool_calls.append({
                        "name": tool_name,
                        "output": output
                    })

                # Check for tool result
                if "Result:" in output and len(tool_calls) > 0:
                    result = output.split("Result:")[1].strip()
                    tool_results.append({
                        "name": tool_calls[-1]["name"],
                        "result": result
                    })

                # Check for final response
                if output.startswith("Assistant:"):
                    final_response = output[len("Assistant:"):].strip()
                    response_received = True

            except queue.Empty:
                pass

        if not response_received:
            logger.error("Timed out waiting for response")
            raise TimeoutError("Timed out waiting for response")

        # Format the response
        response = {
            "content": final_response,
            "role": "assistant"
        }

        # Add tool calls if any
        if tool_calls and tool_results:
            response["has_tool_calls"] = True
            response["tool_calls"] = tool_calls
            response["tool_results"] = tool_results

        return response

    def _read_output(self):
        """Read output from the process in a separate thread"""
        logger.info("Starting output reader thread")
        while self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    logger.debug(f"Process output: {line.strip()}")
                    self.output_queue.put(line.strip())

                # Also check stderr
                while True:
                    err_line = self.process.stderr.readline()
                    if not err_line:
                        break
                    logger.error(f"Process error: {err_line.strip()}")
                    self.output_queue.put(f"ERROR: {err_line.strip()}")
            except Exception as e:
                logger.error(f"Error reading process output: {str(e)}")
                break
        logger.info("Output reader thread stopped")

    def _write_input(self):
        """Write input to the process in a separate thread"""
        logger.info("Starting input writer thread")
        while self.process and self.process.poll() is None:
            try:
                command = self.command_queue.get(timeout=1)
                logger.debug(f"Writing to process: {command}")
                self.process.stdin.write(command + "\n")
                self.process.stdin.flush()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Error writing to process: {str(e)}")
                break
        logger.info("Input writer thread stopped")

    def stop(self):
        """Stop the client process"""
        logger.info("Stopping client process")
        if self.process and self.process.poll() is None:
            try:
                # Try to exit gracefully
                self.command_queue.put("exit")
                time.sleep(1)
                if self.process.poll() is None:
                    logger.info("Terminating process")
                    self.process.terminate()
                    self.process.wait(timeout=5)
            except Exception as e:
                logger.error(f"Error stopping process: {str(e)}")
                # Force kill if needed
                if self.process.poll() is None:
                    logger.info("Killing process")
                    self.process.kill()

        self.process = None
        self.connected = False
        logger.info("Client process stopped")


# Setup page config
st.set_page_config(
    page_title="MCP Client",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'client' not in st.session_state:
    st.session_state.client = MCPClientWrapper()
    st.session_state.tools = []
    st.session_state.connected = False
    st.session_state.messages = []
    st.session_state.processing = False

# Page title
st.title("MCP Client Interface")
st.markdown("Connect to MCP servers and interact with tools using LM Studio's Qwen model.")

# Define sidebar
with st.sidebar:
    st.header("Server Connection")

    client_script = st.text_input(
        "Terminal Client Path",
        value="client_qwen.py",
        help="Path to the terminal MCP client script"
    )

    server_script = st.text_input(
        "Server Script Path",
        help="Path to the MCP server script (e.g., tools.py)"
    )

    connect_col, clear_col = st.columns(2)

    with connect_col:
        connect_button = st.button(
            "Connect",
            disabled=st.session_state.processing,
            use_container_width=True
        )

    with clear_col:
        clear_button = st.button(
            "Clear Chat",
            disabled=st.session_state.processing,
            use_container_width=True
        )

    if connect_button and client_script and server_script:
        with st.spinner("Connecting to server..."):
            try:
                # Verify file paths
                if not os.path.exists(client_script):
                    st.error(f"Client script not found: {client_script}")
                elif not os.path.exists(server_script):
                    st.error(f"Server script not found: {server_script}")
                else:
                    # Connect to the server
                    st.session_state.tools = st.session_state.client.connect(client_script, server_script)
                    st.session_state.connected = True
                    st.success(f"Connected to server: {server_script}")
                    st.session_state.messages = []  # Clear messages on new connection
            except Exception as e:
                st.error(f"Connection failed: {str(e)}")

    if clear_button:
        st.session_state.messages = []
        st.success("Chat history cleared")

    # Show connection status and tools
    st.header("Connection Status")
    if st.session_state.connected:
        st.success("Connected to MCP Server")
        st.info(f"Server: {st.session_state.client.server_script}")

        # Show available tools
        if st.session_state.tools:
            st.header("Available Tools")
            for tool in st.session_state.tools:
                st.write(f"- {tool}")
        else:
            st.warning("No tools available")
    else:
        st.error("Not connected to any server")

# Main chat interface
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑‍💻"):
                st.write(message["content"])
        elif message["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])
        elif message["role"] == "tool":
            with st.chat_message("assistant", avatar="🔧"):
                tool_name = message.get("name", "Tool")
                st.write(f"**{tool_name} Result:**")
                st.code(message["content"])
        elif message["role"] == "system":
            with st.chat_message("assistant", avatar="ℹ️"):
                st.write(message["content"])

# Input area
if st.session_state.connected:
    user_input = st.chat_input(
        "Type your message here...",
        disabled=st.session_state.processing
    )

    if user_input and not st.session_state.processing:
        st.session_state.processing = True

        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        # Create a placeholder for the response
        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            response_placeholder.write("Processing your request...")

        try:
            # Process the query
            response = st.session_state.client.send_query(user_input)

            # Check if there were tool calls
            if "has_tool_calls" in response and response["has_tool_calls"]:
                # Remove the placeholder
                response_placeholder.empty()

                # Display tool calls and results
                for i, tool_call in enumerate(response["tool_calls"]):
                    # Add tool call message
                    tool_message = {
                        "role": "assistant",
                        "content": f"I need to use the '{tool_call['name']}' tool to help with that."
                    }
                    st.session_state.messages.append(tool_message)

                    # Add tool result
                    if i < len(response["tool_results"]):
                        result_message = {
                            "role": "tool",
                            "name": tool_call["name"],
                            "content": response["tool_results"][i]["result"]
                        }
                        st.session_state.messages.append(result_message)

                # Add final assistant response
                assistant_message = {
                    "role": "assistant",
                    "content": response["content"]
                }
                st.session_state.messages.append(assistant_message)
            else:
                # Just update the placeholder with the response
                response_placeholder.empty()

                # Add assistant message
                assistant_message = {
                    "role": "assistant",
                    "content": response["content"]
                }
                st.session_state.messages.append(assistant_message)

        except Exception as e:
            # Display error message
            error_message = f"⚠️ Error: {str(e)}"
            response_placeholder.write(error_message)

            # Add error to chat
            st.session_state.messages.append({
                "role": "system",
                "content": error_message
            })

        st.session_state.processing = False
        st.rerun()
else:
    st.warning("Please connect to a server first using the sidebar.")


# Register cleanup on app shutdown
def cleanup():
    if 'client' in st.session_state and st.session_state.client:
        logger.info("Performing cleanup on app shutdown")
        st.session_state.client.stop()


import atexit

atexit.register(cleanup)