"""
Tools Module - Tool Registry and Built-in Tools for ExecutorAgent

This module provides the tool infrastructure for the Princeps system:
- ToolRegistry: Centralized registry for tool discovery and execution
- BaseTool: Abstract base class for implementing tools
- Built-in tools: Shell, API, File, JSON, and utility tools
- Security: Sandboxing, tenant policies, and permission checks

Usage:
    from tools import ToolRegistry, register_builtin_tools

    # Get the global registry
    registry = get_tool_registry()

    # Register built-in tools with security settings
    register_builtin_tools(
        registry=registry,
        shell_sandbox="/tmp/sandbox",
        allowed_domains={"api.example.com"},
    )

    # Execute a tool
    result = await registry.execute(
        "shell",
        {"command": "ls -la"},
        context={"tenant_id": "tenant_123"},
    )
"""

from framework.tools.tool_registry import (
    # Core classes
    ToolRegistry,
    BaseTool,
    ToolResult,
    ToolDefinition,
    ToolParameter,
    TenantPolicy,
    # Enums
    ToolSecurityLevel,
    ToolCategory,
    ToolExecutionStatus,
    # Utilities
    get_tool_registry,
    set_tool_registry,
    register_tool,
    tool,  # Decorator
)

from framework.tools.builtin_tools import (
    # Shell tool
    ShellTool,
    # Web/API tools
    WebRequestTool,
    # File tools
    FileReadTool,
    FileWriteTool,
    # Utility tools
    JSONTransformTool,
    WaitTool,
    LogTool,
    StoreVariableTool,
    GetVariableTool,
    # Registration function
    register_builtin_tools,
)

__all__ = [
    # Core registry
    "ToolRegistry",
    "BaseTool",
    "ToolResult",
    "ToolDefinition",
    "ToolParameter",
    "TenantPolicy",
    # Enums
    "ToolSecurityLevel",
    "ToolCategory",
    "ToolExecutionStatus",
    # Global registry access
    "get_tool_registry",
    "set_tool_registry",
    "register_tool",
    "tool",
    # Built-in tools
    "ShellTool",
    "WebRequestTool",
    "FileReadTool",
    "FileWriteTool",
    "JSONTransformTool",
    "WaitTool",
    "LogTool",
    "StoreVariableTool",
    "GetVariableTool",
    "register_builtin_tools",
]
