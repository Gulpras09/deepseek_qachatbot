import streamlit as st
import ollama
from datetime import datetime
import json
import re
from typing import Any, Dict, Union
import uuid


# Custom Formatter Class for Structured Log Entry
class LogFormatter:
    def __init__(self, max_field_length: int = 100, padding_char: str = " "):
        self.max_field_length = max_field_length
        self.padding_char = padding_char
        self.type_formats = {
            str: lambda x: str(x),
            int: lambda x: f"{x:d}",
            float: lambda x: f"{x:.2f}",
            bool: lambda x: str(x).lower(),
            dict: lambda x: json.dumps(x, ensure_ascii=False),
            list: lambda x: json.dumps(x, ensure_ascii=False),
        }

    def format_field(self, value: Any, field_name: str, width: int = None) -> str:
        """Format a single field based on its type and constraints."""
        width = width or self.max_field_length
        try:
            # Get formatter for the value's type
            formatter = self.type_formats.get(type(value), str)
            formatted = formatter(value)

            # Truncate if necessary
            if len(formatted) > width:
                formatted = formatted[: width - 3] + "..."

            # Pad to align
            return formatted.ljust(width, self.padding_char)
        except Exception as e:
            return f"Error formatting {field_name}: {str(e)}".ljust(width, self.padding_char)

    def create_log_entry(self, log_data: Dict[str, Any]) -> str:
        """Create a structured log entry from a dictionary of fields."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(uuid.uuid4())[:8]

        # Define field widths for consistent formatting
        field_widths = {
            "timestamp": 20,
            "log_id": 10,
            "user_input": 50,
            "response": 80,
            "model": 15,
            "duration_ms": 10,
        }

        # Prepare formatted fields
        formatted_fields = [
            self.format_field(timestamp, "timestamp", field_widths["timestamp"]),
            self.format_field(log_id, "log_id", field_widths["log_id"]),
        ]

        # Format each provided log data field
        for key, value in log_data.items():
            width = field_widths.get(key, self.max_field_length)
            formatted_fields.append(self.format_field(value, key, width))

        return " | ".join(formatted_fields)


# Streamlit App
st.title("DeepSeek R1:1.5B Chatbot")
st.write("Interact with the DeepSeek R1 model via Ollama, with structured logging.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize logger
logger = LogFormatter(max_field_length=100, padding_char=" ")

# Input form
with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_area("Your message:", height=100)
    submit_button = st.form_submit_button("Send")

# Handle submission
if submit_button and user_input:
    try:
        # Record start time for duration
        start_time = datetime.now()

        # Append user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Call Ollama with DeepSeek model
        response = ollama.chat(
            model="deepseek:1.5b",
            messages=[{"role": "user", "content": user_input}],
        )

        # Calculate duration
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Get model response
        assistant_response = response.get("message", {}).get("content", "No response received")

        # Append assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

        # Create log entry
        log_data = {
            "user_input": user_input,
            "response": assistant_response,
            "model": "deepseek:1.5b",
            "duration_ms": duration_ms,
        }
        log_entry = logger.create_log_entry(log_data)

        # Write log to file
        with open("chat_logs.txt", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        log_data = {
            "user_input": user_input,
            "response": f"Error: {str(e)}",
            "model": "deepseek:1.5b",
            "duration_ms": 0,
        }
        log_entry = logger.create_log_entry(log_data)
        with open("chat_logs.txt", "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

# Display chat history
st.subheader("Chat History")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Display recent logs
st.subheader("Recent Logs")
try:
    with open("chat_logs.txt", "r", encoding="utf-8") as f:
        logs = f.readlines()[-5:]  # Show last 5 log entries
        for log in logs:
            st.text(log.strip())
except FileNotFoundError:
    st.write("No logs available yet.")