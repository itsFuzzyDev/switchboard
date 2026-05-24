#!/usr/bin/env python3
"""Interactive test script for Switchboard + Ollama (gemma4:31b-cloud).
Run with: PYTHONPATH=. python3 test.py
"""

import json, time, threading
from switchboard import Switchboard

# ---------------------------------------------------------------------------
# tool definitions
# ---------------------------------------------------------------------------

def get_weather(location: str) -> str:
    return f"Weather in {location}: Sunny, 22°C, light breeze."

def get_time(timezone: str) -> str:
    from datetime import datetime
    return f"Current time ({timezone}): {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current date and time for a timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "Timezone like UTC, EST, PST"}},
                "required": ["timezone"],
            },
        },
    },
]

TOOL_MAP = {"get_weather": get_weather, "get_time": get_time}

# ---------------------------------------------------------------------------
# tool executor
# ---------------------------------------------------------------------------

def execute_tool(tc: dict) -> str:
    fn_name = tc.get("function", {}).get("name", "")
    raw_args = tc.get("function", {}).get("arguments", "{}")
    if isinstance(raw_args, dict):
        args = raw_args
    else:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except Exception:
            args = {}
    fn = TOOL_MAP.get(fn_name)
    if not fn:
        return f"[error: unknown tool '{fn_name}']"
    try:
        return fn(**args)
    except TypeError as e:
        return f"[error calling {fn_name}: {e}]"
    except Exception as e:
        return f"[error calling {fn_name}: {e}]"

# ---------------------------------------------------------------------------
# streaming printer
# ---------------------------------------------------------------------------

def stream_response(sb, messages, tools=None):
    """Stream a chat response and return the full assistant message dict."""
    assistant_msg = {"role": "assistant", "content": "", "tool_calls": []}
    first_token = True
    spinner_active = True
    saw_reasoning = False

    def _spinner():
        dots = ["", ".", "..", "..."]
        i = 0
        while spinner_active:
            print(f"\r\033[KAssistant: {dots[i % len(dots)]}", end="", flush=True)
            time.sleep(0.4)
            i += 1

    t = threading.Thread(target=_spinner, daemon=True)
    t.start()

    for chunk in sb.chat(messages=messages, stream=True, tools=tools, thinking=True):
        msg = chunk.get("message", {})

        if first_token:
            spinner_active = False
            print("\r\033[K", end="", flush=True)
            first_token = False

        # Show reasoning if the provider sent it
        reasoning = chunk.get("reasoning_content")
        if reasoning is not None and str(reasoning).strip():
            saw_reasoning = True
            print(f"\033[93m[thinking] {reasoning}\033[0m", end="", flush=True)

        content = msg.get("content", "")
        if content:
            print(content, end="", flush=True)
            assistant_msg["content"] += content

        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {}).get("name", "unknown")
                print(f"\n  [calling tool: {fn}]\n", end="", flush=True)
            assistant_msg["tool_calls"].extend(tool_calls)

    if first_token:
        spinner_active = False
        print("\r\033[K", end="", flush=True)

    # If we expected thinking but got none, say so
    if not saw_reasoning and not assistant_msg["content"] and not assistant_msg["tool_calls"]:
        print("(model returned no content or thinking)")

    print()
    return assistant_msg

# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------

def main():
    sb = Switchboard(provider="ollama", model="gemma4:31b-cloud")

    messages = [
        {"role": "system", "content": "You are a helpful assistant. When the user asks about weather or time, use the provided tools."}
    ]

    print("=== Switchboard Test — Ollama (gemma4:31b-cloud) ===")
    print("NOTE: This model may not expose thinking through Ollama's API.")
    print("      If you want to see reasoning, try: ollama pull deepseek-r1")
    print("Type your question below. Type 'quit' to exit.\n")

    while True:
        user_text = input("You: ").strip()
        if user_text.lower() in ("quit", "exit", "q"):
            break

        messages.append({"role": "user", "content": user_text})

        # first pass
        assistant_msg = stream_response(sb, messages, tools=TOOLS)
        messages.append(assistant_msg)

        # handle tool calls
        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            continue

        for tc in tool_calls:
            result = execute_tool(tc)
            fn_name = tc.get("function", {}).get("name", "unknown")
            print(f"  [tool result: {fn_name} -> {result}]")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", ""),
                "content": result,
            })

        # second pass after tools
        print("Assistant: ", end="", flush=True)
        assistant_msg = stream_response(sb, messages, tools=TOOLS)
        messages.append(assistant_msg)

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
