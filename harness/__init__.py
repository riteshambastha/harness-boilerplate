"""harness-starter: a minimal harness around an LLM.

The five layers, each in its own module:
  model.py    - provider-agnostic model interface
  agent.py    - the agent loop
  tools/      - tool registry and example tools
  context.py  - what the model sees each step
  logging.py / feedback.py - traces and human feedback
"""

from harness.agent import Agent, AgentResult
from harness.context import ContextManager
from harness.model import Message, ModelResponse, ScriptedClient, ToolCall, client_from_env
from harness.tools import ToolRegistry, default_tools, tool

__all__ = [
    "Agent",
    "AgentResult",
    "ContextManager",
    "Message",
    "ModelResponse",
    "ScriptedClient",
    "ToolCall",
    "ToolRegistry",
    "client_from_env",
    "default_tools",
    "tool",
]
