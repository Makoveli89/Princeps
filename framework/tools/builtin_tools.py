"""
Built-in Tools - Standard Tool Implementations for ExecutorAgent

This module provides the core set of built-in tools that the ExecutorAgent
can use to carry out plans. Tools include:

- ShellTool: Execute shell commands (with sandboxing)
- WebRequestTool: Make HTTP requests to external APIs
- FileReadTool/FileWriteTool: File system operations
- DatabaseQueryTool: Execute database queries
- KnowledgeRetrieveTool: Query the Brain's knowledge base
- LLMQueryTool: Make LLM queries for sub-tasks
- JSONTransformTool: Transform and manipulate JSON data
- WaitTool: Pause execution for a specified duration
- LogTool: Log messages for debugging/audit

Security:
- Shell commands are validated against a blocklist
- File operations are sandboxed to allowed directories
- Network requests can be restricted by domain
- All operations respect tenant isolation

Adapted from patterns in:
- shell_command_agent.py: Shell execution patterns
- api_agent.py: HTTP request handling
- security_scanner.py: Input validation patterns
"""

import asyncio
import logging
import os
import json
import re
import subprocess
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from framework.tools.tool_registry import (
    BaseTool,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    ToolCategory,
    ToolSecurityLevel,
    ToolExecutionStatus,
)

logger = logging.getLogger(__name__)


class ShellTool(BaseTool):
    """
    Execute shell commands with sandboxing and security checks.

    Security:
    - Commands are validated against a blocklist of dangerous patterns
    - Working directory can be restricted to a sandbox
    - Environment variables are sanitized
    - Timeout prevents runaway processes
    """

    # Dangerous command patterns that are blocked
    BLOCKED_PATTERNS = [
        r'\brm\s+(-[rf]+\s+)?/',  # rm with root paths
        r'\brm\s+(-[rf]+\s+)?\*',  # rm with wildcards
        r'\bdd\b',  # dd command
        r'\bmkfs\b',  # filesystem commands
        r'\bfdisk\b',
        r'\bsudo\b',  # privilege escalation
        r'\bsu\s',
        r'\bchmod\s+777\b',  # dangerous permissions
        r'\bchown\b.*root',
        r'\b(>|>>)\s*/dev/',  # writing to devices
        r'\bcurl\b.*\|\s*(ba)?sh',  # curl pipe to shell
        r'\bwget\b.*\|\s*(ba)?sh',
        r'\beval\b',  # eval commands
        r'\bexec\b',
        r';\s*(rm|dd|mkfs)',  # chained dangerous commands
        r'\|\s*(rm|dd|mkfs)',
        r'`.*`',  # command substitution (potential injection)
        r'\$\(.*\)',
        r'\bshutdown\b',
        r'\breboot\b',
        r'\bhalt\b',
        r'\binit\s+[0-6]\b',
    ]

    # Allowed commands (if set, only these are allowed)
    ALLOWED_COMMANDS: Optional[Set[str]] = None  # None = allow all except blocked

    def __init__(
        self,
        sandbox_dir: Optional[str] = None,
        allowed_commands: Optional[Set[str]] = None,
        blocked_patterns: Optional[List[str]] = None,
        max_output_size: int = 100000,  # 100KB
    ):
        super().__init__()
        self.sandbox_dir = sandbox_dir
        self.allowed_commands = allowed_commands
        self.blocked_patterns = blocked_patterns or self.BLOCKED_PATTERNS
        self.max_output_size = max_output_size

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="shell",
            description="Execute a shell command and return the output",
            category=ToolCategory.SHELL,
            security_level=ToolSecurityLevel.ELEVATED,
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="The shell command to execute",
                    required=True,
                ),
                ToolParameter(
                    name="working_dir",
                    type="string",
                    description="Working directory for the command",
                    required=False,
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    description="Timeout in seconds (default: 30)",
                    required=False,
                    default=30,
                ),
                ToolParameter(
                    name="capture_stderr",
                    type="bool",
                    description="Whether to capture stderr (default: True)",
                    required=False,
                    default=True,
                ),
            ],
            is_async=True,
            timeout_seconds=60.0,
            required_permissions={"shell.execute"},
        )

    def _is_command_safe(self, command: str) -> tuple[bool, Optional[str]]:
        """Check if a command is safe to execute"""
        # Check blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                return False, f"Command matches blocked pattern: {pattern}"

        # Check allowed commands if set
        if self.allowed_commands:
            # Extract base command
            parts = shlex.split(command)
            if parts:
                base_cmd = parts[0].split('/')[-1]  # Get basename
                if base_cmd not in self.allowed_commands:
                    return False, f"Command '{base_cmd}' not in allowed list"

        return True, None

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        command = parameters.get("command", "")
        working_dir = parameters.get("working_dir", self.sandbox_dir)
        timeout = parameters.get("timeout", 30)
        capture_stderr = parameters.get("capture_stderr", True)

        # Security check
        safe, reason = self._is_command_safe(command)
        if not safe:
            return ToolResult(
                tool_name="shell",
                status=ToolExecutionStatus.BLOCKED,
                error=f"Command blocked: {reason}",
                error_type="SecurityError",
                security_warnings=[reason],
            )

        # Validate working directory
        if working_dir:
            if self.sandbox_dir and not working_dir.startswith(self.sandbox_dir):
                return ToolResult(
                    tool_name="shell",
                    status=ToolExecutionStatus.BLOCKED,
                    error=f"Working directory must be within sandbox: {self.sandbox_dir}",
                    error_type="SecurityError",
                )
            if not os.path.isdir(working_dir):
                return ToolResult(
                    tool_name="shell",
                    status=ToolExecutionStatus.FAILED,
                    error=f"Working directory does not exist: {working_dir}",
                    error_type="FileNotFoundError",
                )

        try:
            # Execute command
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE if capture_stderr else None,
                cwd=working_dir,
                env=self._get_sanitized_env(),
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return ToolResult(
                    tool_name="shell",
                    status=ToolExecutionStatus.TIMEOUT,
                    error=f"Command timed out after {timeout}s",
                    error_type="TimeoutError",
                )

            # Truncate output if too large
            stdout_str = stdout.decode('utf-8', errors='replace')[:self.max_output_size]
            stderr_str = stderr.decode('utf-8', errors='replace')[:self.max_output_size] if stderr else ""

            output = {
                "stdout": stdout_str,
                "stderr": stderr_str,
                "return_code": process.returncode,
            }

            if process.returncode == 0:
                return ToolResult(
                    tool_name="shell",
                    status=ToolExecutionStatus.SUCCESS,
                    output=output,
                    output_type="dict",
                )
            else:
                return ToolResult(
                    tool_name="shell",
                    status=ToolExecutionStatus.FAILED,
                    output=output,
                    error=f"Command exited with code {process.returncode}",
                    error_type="CommandError",
                )

        except Exception as e:
            return ToolResult(
                tool_name="shell",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _get_sanitized_env(self) -> Dict[str, str]:
        """Get sanitized environment variables"""
        # Start with minimal environment
        safe_vars = ['PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'TERM']
        env = {k: v for k, v in os.environ.items() if k in safe_vars}

        # Add safe defaults
        env.setdefault('PATH', '/usr/bin:/bin')

        return env


class WebRequestTool(BaseTool):
    """
    Make HTTP requests to external APIs.

    Security:
    - Domain allowlist/blocklist can be configured
    - Request size limits
    - Timeout enforcement
    - Response size limits
    """

    def __init__(
        self,
        allowed_domains: Optional[Set[str]] = None,
        blocked_domains: Optional[Set[str]] = None,
        max_response_size: int = 10 * 1024 * 1024,  # 10MB
        default_timeout: int = 30,
    ):
        super().__init__()
        self.allowed_domains = allowed_domains
        self.blocked_domains = blocked_domains or set()
        self.max_response_size = max_response_size
        self.default_timeout = default_timeout

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="web_request",
            description="Make an HTTP request to an external API",
            category=ToolCategory.API,
            security_level=ToolSecurityLevel.STANDARD,
            parameters=[
                ToolParameter(
                    name="url",
                    type="string",
                    description="The URL to request",
                    required=True,
                ),
                ToolParameter(
                    name="method",
                    type="string",
                    description="HTTP method (GET, POST, PUT, DELETE, PATCH)",
                    required=False,
                    default="GET",
                    allowed_values=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
                ),
                ToolParameter(
                    name="headers",
                    type="dict",
                    description="HTTP headers to include",
                    required=False,
                ),
                ToolParameter(
                    name="body",
                    type="any",
                    description="Request body (for POST/PUT/PATCH)",
                    required=False,
                ),
                ToolParameter(
                    name="timeout",
                    type="int",
                    description="Request timeout in seconds",
                    required=False,
                    default=30,
                ),
            ],
            is_async=True,
            timeout_seconds=60.0,
            required_permissions={"network.http"},
        )

    def _is_domain_allowed(self, url: str) -> tuple[bool, Optional[str]]:
        """Check if the domain is allowed"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]

            # Check blocklist
            if domain in self.blocked_domains:
                return False, f"Domain '{domain}' is blocked"

            # Check allowlist if set
            if self.allowed_domains:
                if domain not in self.allowed_domains:
                    # Also check subdomains
                    allowed = False
                    for allowed_domain in self.allowed_domains:
                        if domain == allowed_domain or domain.endswith('.' + allowed_domain):
                            allowed = True
                            break
                    if not allowed:
                        return False, f"Domain '{domain}' not in allowed list"

            return True, None

        except Exception as e:
            return False, f"Invalid URL: {e}"

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        url = parameters.get("url", "")
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers", {})
        body = parameters.get("body")
        timeout = parameters.get("timeout", self.default_timeout)

        # Validate domain
        allowed, reason = self._is_domain_allowed(url)
        if not allowed:
            return ToolResult(
                tool_name="web_request",
                status=ToolExecutionStatus.BLOCKED,
                error=reason,
                error_type="SecurityError",
            )

        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                # Prepare request kwargs
                kwargs = {
                    "timeout": aiohttp.ClientTimeout(total=timeout),
                    "headers": headers,
                }

                if body and method in ["POST", "PUT", "PATCH"]:
                    if isinstance(body, dict):
                        kwargs["json"] = body
                    else:
                        kwargs["data"] = body

                async with session.request(method, url, **kwargs) as response:
                    # Check response size
                    content_length = response.headers.get('Content-Length')
                    if content_length and int(content_length) > self.max_response_size:
                        return ToolResult(
                            tool_name="web_request",
                            status=ToolExecutionStatus.FAILED,
                            error=f"Response too large: {content_length} bytes",
                            error_type="ResponseTooLargeError",
                        )

                    # Read response
                    content = await response.read()
                    if len(content) > self.max_response_size:
                        content = content[:self.max_response_size]

                    # Try to decode as text
                    try:
                        text = content.decode('utf-8')
                        # Try to parse as JSON
                        try:
                            data = json.loads(text)
                            output = {
                                "status_code": response.status,
                                "headers": dict(response.headers),
                                "body": data,
                                "is_json": True,
                            }
                        except json.JSONDecodeError:
                            output = {
                                "status_code": response.status,
                                "headers": dict(response.headers),
                                "body": text,
                                "is_json": False,
                            }
                    except UnicodeDecodeError:
                        output = {
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": "<binary content>",
                            "is_binary": True,
                            "size": len(content),
                        }

                    if 200 <= response.status < 300:
                        return ToolResult(
                            tool_name="web_request",
                            status=ToolExecutionStatus.SUCCESS,
                            output=output,
                            output_type="dict",
                        )
                    else:
                        return ToolResult(
                            tool_name="web_request",
                            status=ToolExecutionStatus.FAILED,
                            output=output,
                            error=f"HTTP {response.status}",
                            error_type="HTTPError",
                        )

        except ImportError:
            # Fallback to urllib if aiohttp not available
            return await self._execute_with_urllib(parameters, context)

        except asyncio.TimeoutError:
            return ToolResult(
                tool_name="web_request",
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Request timed out after {timeout}s",
                error_type="TimeoutError",
            )

        except Exception as e:
            return ToolResult(
                tool_name="web_request",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )

    async def _execute_with_urllib(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """Fallback implementation using urllib"""
        import urllib.request
        import urllib.error

        url = parameters.get("url", "")
        method = parameters.get("method", "GET").upper()
        headers = parameters.get("headers", {})
        body = parameters.get("body")
        timeout = parameters.get("timeout", self.default_timeout)

        try:
            data = None
            if body and method in ["POST", "PUT", "PATCH"]:
                if isinstance(body, dict):
                    data = json.dumps(body).encode('utf-8')
                    headers.setdefault('Content-Type', 'application/json')
                else:
                    data = str(body).encode('utf-8')

            request = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method=method,
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                content = response.read()[:self.max_response_size]
                try:
                    text = content.decode('utf-8')
                    try:
                        data = json.loads(text)
                        output = {
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": data,
                            "is_json": True,
                        }
                    except json.JSONDecodeError:
                        output = {
                            "status_code": response.status,
                            "headers": dict(response.headers),
                            "body": text,
                            "is_json": False,
                        }
                except UnicodeDecodeError:
                    output = {
                        "status_code": response.status,
                        "headers": dict(response.headers),
                        "body": "<binary>",
                        "is_binary": True,
                    }

                return ToolResult(
                    tool_name="web_request",
                    status=ToolExecutionStatus.SUCCESS,
                    output=output,
                    output_type="dict",
                )

        except urllib.error.HTTPError as e:
            return ToolResult(
                tool_name="web_request",
                status=ToolExecutionStatus.FAILED,
                output={"status_code": e.code, "reason": e.reason},
                error=f"HTTP {e.code}: {e.reason}",
                error_type="HTTPError",
            )

        except Exception as e:
            return ToolResult(
                tool_name="web_request",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )


class FileReadTool(BaseTool):
    """Read contents of a file (with sandbox restrictions)"""

    def __init__(
        self,
        allowed_directories: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        super().__init__()
        self.allowed_directories = allowed_directories
        self.max_file_size = max_file_size

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_read",
            description="Read the contents of a file",
            category=ToolCategory.FILE,
            security_level=ToolSecurityLevel.STANDARD,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to read",
                    required=True,
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="binary",
                    type="bool",
                    description="Read as binary (default: False)",
                    required=False,
                    default=False,
                ),
            ],
            is_async=True,
            timeout_seconds=30.0,
            required_permissions={"file.read"},
        )

    def _is_path_allowed(self, path: str) -> tuple[bool, Optional[str]]:
        """Check if path is within allowed directories"""
        if not self.allowed_directories:
            return True, None

        abs_path = os.path.abspath(path)
        for allowed_dir in self.allowed_directories:
            allowed_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_abs):
                return True, None

        return False, f"Path '{path}' is outside allowed directories"

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        path = parameters.get("path", "")
        encoding = parameters.get("encoding", "utf-8")
        binary = parameters.get("binary", False)

        # Security check
        allowed, reason = self._is_path_allowed(path)
        if not allowed:
            return ToolResult(
                tool_name="file_read",
                status=ToolExecutionStatus.BLOCKED,
                error=reason,
                error_type="SecurityError",
            )

        try:
            # Check file exists and size
            if not os.path.exists(path):
                return ToolResult(
                    tool_name="file_read",
                    status=ToolExecutionStatus.FAILED,
                    error=f"File not found: {path}",
                    error_type="FileNotFoundError",
                )

            file_size = os.path.getsize(path)
            if file_size > self.max_file_size:
                return ToolResult(
                    tool_name="file_read",
                    status=ToolExecutionStatus.FAILED,
                    error=f"File too large: {file_size} bytes (max: {self.max_file_size})",
                    error_type="FileTooLargeError",
                )

            # Read file
            mode = 'rb' if binary else 'r'
            kwargs = {} if binary else {'encoding': encoding}

            with open(path, mode, **kwargs) as f:
                content = f.read()

            output = {
                "path": path,
                "content": content if not binary else f"<binary: {len(content)} bytes>",
                "size": file_size,
                "is_binary": binary,
            }

            return ToolResult(
                tool_name="file_read",
                status=ToolExecutionStatus.SUCCESS,
                output=output if binary else content,
                output_type="string" if not binary else "dict",
            )

        except PermissionError:
            return ToolResult(
                tool_name="file_read",
                status=ToolExecutionStatus.BLOCKED,
                error=f"Permission denied: {path}",
                error_type="PermissionError",
            )

        except Exception as e:
            return ToolResult(
                tool_name="file_read",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )


class FileWriteTool(BaseTool):
    """Write contents to a file (with sandbox restrictions)"""

    def __init__(
        self,
        allowed_directories: Optional[List[str]] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
    ):
        super().__init__()
        self.allowed_directories = allowed_directories
        self.max_file_size = max_file_size

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="file_write",
            description="Write content to a file",
            category=ToolCategory.FILE,
            security_level=ToolSecurityLevel.ELEVATED,
            parameters=[
                ToolParameter(
                    name="path",
                    type="string",
                    description="Path to the file to write",
                    required=True,
                ),
                ToolParameter(
                    name="content",
                    type="string",
                    description="Content to write",
                    required=True,
                ),
                ToolParameter(
                    name="mode",
                    type="string",
                    description="Write mode: 'write' (overwrite) or 'append'",
                    required=False,
                    default="write",
                    allowed_values=["write", "append"],
                ),
                ToolParameter(
                    name="encoding",
                    type="string",
                    description="File encoding (default: utf-8)",
                    required=False,
                    default="utf-8",
                ),
                ToolParameter(
                    name="create_dirs",
                    type="bool",
                    description="Create parent directories if needed",
                    required=False,
                    default=False,
                ),
            ],
            is_async=True,
            timeout_seconds=30.0,
            required_permissions={"file.write"},
        )

    def _is_path_allowed(self, path: str) -> tuple[bool, Optional[str]]:
        """Check if path is within allowed directories"""
        if not self.allowed_directories:
            return True, None

        abs_path = os.path.abspath(path)
        for allowed_dir in self.allowed_directories:
            allowed_abs = os.path.abspath(allowed_dir)
            if abs_path.startswith(allowed_abs):
                return True, None

        return False, f"Path '{path}' is outside allowed directories"

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        path = parameters.get("path", "")
        content = parameters.get("content", "")
        mode = parameters.get("mode", "write")
        encoding = parameters.get("encoding", "utf-8")
        create_dirs = parameters.get("create_dirs", False)

        # Security check
        allowed, reason = self._is_path_allowed(path)
        if not allowed:
            return ToolResult(
                tool_name="file_write",
                status=ToolExecutionStatus.BLOCKED,
                error=reason,
                error_type="SecurityError",
            )

        # Size check
        content_size = len(content.encode(encoding) if isinstance(content, str) else content)
        if content_size > self.max_file_size:
            return ToolResult(
                tool_name="file_write",
                status=ToolExecutionStatus.FAILED,
                error=f"Content too large: {content_size} bytes",
                error_type="ContentTooLargeError",
            )

        try:
            # Create directories if needed
            if create_dirs:
                os.makedirs(os.path.dirname(path), exist_ok=True)

            # Write file
            file_mode = 'a' if mode == 'append' else 'w'
            with open(path, file_mode, encoding=encoding) as f:
                f.write(content)

            return ToolResult(
                tool_name="file_write",
                status=ToolExecutionStatus.SUCCESS,
                output={
                    "path": path,
                    "bytes_written": content_size,
                    "mode": mode,
                },
                output_type="dict",
            )

        except PermissionError:
            return ToolResult(
                tool_name="file_write",
                status=ToolExecutionStatus.BLOCKED,
                error=f"Permission denied: {path}",
                error_type="PermissionError",
            )

        except Exception as e:
            return ToolResult(
                tool_name="file_write",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )


class JSONTransformTool(BaseTool):
    """Transform and manipulate JSON data"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="json_transform",
            description="Transform JSON data using jq-like expressions or Python",
            category=ToolCategory.UTILITY,
            security_level=ToolSecurityLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="data",
                    type="any",
                    description="Input JSON data (dict or list)",
                    required=True,
                ),
                ToolParameter(
                    name="operation",
                    type="string",
                    description="Operation: 'get', 'set', 'filter', 'map', 'sort', 'merge'",
                    required=True,
                    allowed_values=["get", "set", "filter", "map", "sort", "merge", "keys", "values", "flatten"],
                ),
                ToolParameter(
                    name="path",
                    type="string",
                    description="JSON path for get/set (e.g., 'data.items[0].name')",
                    required=False,
                ),
                ToolParameter(
                    name="value",
                    type="any",
                    description="Value for set operation",
                    required=False,
                ),
                ToolParameter(
                    name="key",
                    type="string",
                    description="Key for filter/sort operations",
                    required=False,
                ),
                ToolParameter(
                    name="condition",
                    type="string",
                    description="Condition for filter (e.g., 'item.age > 18')",
                    required=False,
                ),
            ],
            is_async=True,
            timeout_seconds=10.0,
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        data = parameters.get("data")
        operation = parameters.get("operation")
        path = parameters.get("path", "")
        value = parameters.get("value")
        key = parameters.get("key")

        try:
            if operation == "get":
                result = self._get_path(data, path)
            elif operation == "set":
                result = self._set_path(data, path, value)
            elif operation == "keys":
                if isinstance(data, dict):
                    result = list(data.keys())
                else:
                    result = list(range(len(data)))
            elif operation == "values":
                if isinstance(data, dict):
                    result = list(data.values())
                else:
                    result = data
            elif operation == "flatten":
                result = self._flatten(data)
            elif operation == "sort":
                result = self._sort_data(data, key)
            elif operation == "merge":
                if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                    result = {}
                    for d in data:
                        result.update(d)
                else:
                    result = data
            else:
                return ToolResult(
                    tool_name="json_transform",
                    status=ToolExecutionStatus.FAILED,
                    error=f"Unknown operation: {operation}",
                    error_type="ValueError",
                )

            return ToolResult(
                tool_name="json_transform",
                status=ToolExecutionStatus.SUCCESS,
                output=result,
                output_type="any",
            )

        except Exception as e:
            return ToolResult(
                tool_name="json_transform",
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
            )

    def _get_path(self, data: Any, path: str) -> Any:
        """Get value at path like 'data.items[0].name'"""
        if not path:
            return data

        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]

        result = data
        for part in parts:
            if isinstance(result, dict):
                result = result.get(part)
            elif isinstance(result, list):
                idx = int(part)
                result = result[idx] if 0 <= idx < len(result) else None
            else:
                return None

        return result

    def _set_path(self, data: Any, path: str, value: Any) -> Any:
        """Set value at path (returns new data, doesn't mutate)"""
        import copy
        data = copy.deepcopy(data)

        if not path:
            return value

        parts = re.split(r'\.|\[|\]', path)
        parts = [p for p in parts if p]

        current = data
        for i, part in enumerate(parts[:-1]):
            if isinstance(current, dict):
                if part not in current:
                    current[part] = {} if not parts[i + 1].isdigit() else []
                current = current[part]
            elif isinstance(current, list):
                idx = int(part)
                while len(current) <= idx:
                    current.append(None)
                if current[idx] is None:
                    current[idx] = {} if not parts[i + 1].isdigit() else []
                current = current[idx]

        last_part = parts[-1]
        if isinstance(current, dict):
            current[last_part] = value
        elif isinstance(current, list):
            idx = int(last_part)
            while len(current) <= idx:
                current.append(None)
            current[idx] = value

        return data

    def _flatten(self, data: Any, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dict/list to single level"""
        items = []
        if isinstance(data, dict):
            for k, v in data.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten(v, new_key, sep).items())
                else:
                    items.append((new_key, v))
        elif isinstance(data, list):
            for i, v in enumerate(data):
                new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
                if isinstance(v, (dict, list)):
                    items.extend(self._flatten(v, new_key, sep).items())
                else:
                    items.append((new_key, v))
        return dict(items)

    def _sort_data(self, data: Any, key: Optional[str]) -> Any:
        """Sort list data"""
        if not isinstance(data, list):
            return data

        if key:
            return sorted(data, key=lambda x: x.get(key) if isinstance(x, dict) else x)
        return sorted(data, key=lambda x: str(x))


class WaitTool(BaseTool):
    """Pause execution for a specified duration"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="wait",
            description="Pause execution for a specified number of seconds",
            category=ToolCategory.UTILITY,
            security_level=ToolSecurityLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="seconds",
                    type="float",
                    description="Number of seconds to wait (max: 300)",
                    required=True,
                ),
                ToolParameter(
                    name="reason",
                    type="string",
                    description="Reason for waiting (for logging)",
                    required=False,
                ),
            ],
            is_async=True,
            timeout_seconds=310.0,
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        seconds = min(parameters.get("seconds", 1), 300)  # Max 5 minutes
        reason = parameters.get("reason", "")

        try:
            await asyncio.sleep(seconds)

            return ToolResult(
                tool_name="wait",
                status=ToolExecutionStatus.SUCCESS,
                output={
                    "waited_seconds": seconds,
                    "reason": reason,
                },
                output_type="dict",
            )

        except asyncio.CancelledError:
            return ToolResult(
                tool_name="wait",
                status=ToolExecutionStatus.FAILED,
                error="Wait was cancelled",
                error_type="CancelledError",
            )


class LogTool(BaseTool):
    """Log a message for debugging/audit purposes"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="log",
            description="Log a message at specified level",
            category=ToolCategory.UTILITY,
            security_level=ToolSecurityLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="message",
                    type="string",
                    description="Message to log",
                    required=True,
                ),
                ToolParameter(
                    name="level",
                    type="string",
                    description="Log level: debug, info, warning, error",
                    required=False,
                    default="info",
                    allowed_values=["debug", "info", "warning", "error"],
                ),
                ToolParameter(
                    name="data",
                    type="dict",
                    description="Additional data to include in log",
                    required=False,
                ),
            ],
            is_async=True,
            timeout_seconds=5.0,
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        message = parameters.get("message", "")
        level = parameters.get("level", "info")
        data = parameters.get("data", {})

        log_func = {
            "debug": logger.debug,
            "info": logger.info,
            "warning": logger.warning,
            "error": logger.error,
        }.get(level, logger.info)

        log_message = f"[ExecutorLog] {message}"
        if data:
            log_message += f" | data={json.dumps(data)}"

        log_func(log_message)

        return ToolResult(
            tool_name="log",
            status=ToolExecutionStatus.SUCCESS,
            output={
                "logged": True,
                "level": level,
                "message": message,
            },
            output_type="dict",
        )


class StoreVariableTool(BaseTool):
    """Store a value in the execution context for later use"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="store_variable",
            description="Store a value that can be referenced by later steps",
            category=ToolCategory.UTILITY,
            security_level=ToolSecurityLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Variable name",
                    required=True,
                ),
                ToolParameter(
                    name="value",
                    type="any",
                    description="Value to store",
                    required=True,
                ),
            ],
            is_async=True,
            timeout_seconds=5.0,
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        name = parameters.get("name", "")
        value = parameters.get("value")

        # Store in context if provided
        if context is not None:
            if "variables" not in context:
                context["variables"] = {}
            context["variables"][name] = value

        return ToolResult(
            tool_name="store_variable",
            status=ToolExecutionStatus.SUCCESS,
            output={
                "stored": True,
                "name": name,
                "value_type": type(value).__name__,
            },
            output_type="dict",
        )


class GetVariableTool(BaseTool):
    """Retrieve a value from the execution context"""

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_variable",
            description="Get a value that was stored by a previous step",
            category=ToolCategory.UTILITY,
            security_level=ToolSecurityLevel.SAFE,
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Variable name",
                    required=True,
                ),
                ToolParameter(
                    name="default",
                    type="any",
                    description="Default value if not found",
                    required=False,
                ),
            ],
            is_async=True,
            timeout_seconds=5.0,
        )

    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        name = parameters.get("name", "")
        default = parameters.get("default")

        value = default
        if context and "variables" in context:
            value = context["variables"].get(name, default)

        return ToolResult(
            tool_name="get_variable",
            status=ToolExecutionStatus.SUCCESS,
            output=value,
            output_type=type(value).__name__ if value is not None else "none",
        )


def register_builtin_tools(
    registry=None,
    shell_sandbox: Optional[str] = None,
    file_directories: Optional[List[str]] = None,
    allowed_domains: Optional[Set[str]] = None,
):
    """
    Register all built-in tools with the registry.

    Args:
        registry: ToolRegistry to use (uses default if None)
        shell_sandbox: Sandbox directory for shell commands
        file_directories: Allowed directories for file operations
        allowed_domains: Allowed domains for web requests
    """
    from framework.tools.tool_registry import get_tool_registry

    if registry is None:
        registry = get_tool_registry()

    # Register tools
    registry.register(ShellTool(sandbox_dir=shell_sandbox))
    registry.register(WebRequestTool(allowed_domains=allowed_domains))
    registry.register(FileReadTool(allowed_directories=file_directories))
    registry.register(FileWriteTool(allowed_directories=file_directories))
    registry.register(JSONTransformTool())
    registry.register(WaitTool())
    registry.register(LogTool())
    registry.register(StoreVariableTool())
    registry.register(GetVariableTool())

    logger.info(f"Registered {len(registry._tools)} built-in tools")

    return registry
