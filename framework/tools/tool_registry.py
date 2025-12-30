"""
Tool Registry - Centralized Registry of Available Tools/Actions

This module implements a centralized registry for tools that the ExecutorAgent
can invoke. It provides:
- Tool registration and discovery
- Permission and security checks
- Tenant-scoped tool availability
- Tool metadata and documentation
- Execution sandboxing
- Input/output validation

Strategic Intent:
The ToolRegistry provides a clean separation between what tools exist and how
they are invoked. The ExecutorAgent looks up tools by name and invokes them
through a consistent interface. This allows adding new capabilities without
modifying the Executor logic.

Security:
- Each tool has a security level and required permissions
- Tenant policies can disable specific tools
- Dangerous operations require explicit approval
- All tool invocations are logged for audit

Adapted from patterns in:
- base_agent.py: Error handling and logging patterns
- security_scanner.py: Security checking patterns
- tenant_isolation.py: Tenant-scoped operations
"""

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union
import inspect
import functools

logger = logging.getLogger(__name__)


class ToolSecurityLevel(Enum):
    """Security level for tools - determines required permissions"""
    SAFE = "safe"  # No side effects, read-only
    STANDARD = "standard"  # Normal operations with controlled side effects
    ELEVATED = "elevated"  # Requires additional authorization
    DANGEROUS = "dangerous"  # System-level, requires explicit approval
    DISABLED = "disabled"  # Tool is disabled


class ToolCategory(Enum):
    """Category of tool for organization and filtering"""
    SHELL = "shell"  # Shell/command execution
    API = "api"  # External API calls
    DATABASE = "database"  # Database operations
    FILE = "file"  # File system operations
    NETWORK = "network"  # Network operations
    COMPUTE = "compute"  # Computation/processing
    KNOWLEDGE = "knowledge"  # Knowledge base operations
    LLM = "llm"  # LLM-related operations
    UTILITY = "utility"  # General utilities


class ToolExecutionStatus(Enum):
    """Status of a tool execution"""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Blocked by security
    SKIPPED = "skipped"  # Skipped due to conditions
    PENDING = "pending"


@dataclass
class ToolParameter:
    """Definition of a tool parameter"""
    name: str
    type: str  # "string", "int", "float", "bool", "dict", "list", "any"
    description: str
    required: bool = True
    default: Any = None
    allowed_values: Optional[List[Any]] = None
    validation_regex: Optional[str] = None

    def validate(self, value: Any) -> tuple[bool, Optional[str]]:
        """Validate a parameter value"""
        import re

        # Check required
        if value is None:
            if self.required:
                return False, f"Required parameter '{self.name}' is missing"
            return True, None

        # Check allowed values
        if self.allowed_values and value not in self.allowed_values:
            return False, f"Value '{value}' not in allowed values: {self.allowed_values}"

        # Check regex pattern
        if self.validation_regex and isinstance(value, str):
            if not re.match(self.validation_regex, value):
                return False, f"Value '{value}' does not match pattern: {self.validation_regex}"

        # Type validation
        type_map = {
            "string": str,
            "int": int,
            "float": (int, float),
            "bool": bool,
            "dict": dict,
            "list": list,
        }
        if self.type in type_map:
            expected = type_map[self.type]
            if not isinstance(value, expected):
                return False, f"Expected {self.type}, got {type(value).__name__}"

        return True, None


@dataclass
class ToolResult:
    """Result of a tool execution"""
    tool_name: str
    status: ToolExecutionStatus
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Execution metadata
    execution_time_ms: float = 0.0
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    # Security/audit info
    was_sandboxed: bool = False
    security_warnings: List[str] = field(default_factory=list)

    # For chaining
    can_chain: bool = True
    output_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status.value,
            "output": self.output if not isinstance(self.output, bytes) else "<binary>",
            "error": self.error,
            "error_type": self.error_type,
            "execution_time_ms": self.execution_time_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "was_sandboxed": self.was_sandboxed,
            "security_warnings": self.security_warnings,
            "can_chain": self.can_chain,
            "output_type": self.output_type,
        }


@dataclass
class ToolDefinition:
    """Complete definition of a tool"""
    name: str
    description: str
    category: ToolCategory
    security_level: ToolSecurityLevel

    # Parameters
    parameters: List[ToolParameter] = field(default_factory=list)

    # Behavior
    is_async: bool = False
    timeout_seconds: float = 60.0
    max_retries: int = 1

    # Security
    required_permissions: Set[str] = field(default_factory=set)
    blocked_tenants: Set[str] = field(default_factory=set)
    requires_approval: bool = False

    # Output
    output_type: str = "any"  # Expected output type

    # Metadata
    version: str = "1.0.0"
    author: Optional[str] = None
    documentation_url: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "security_level": self.security_level.value,
            "parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.parameters
            ],
            "is_async": self.is_async,
            "timeout_seconds": self.timeout_seconds,
            "required_permissions": list(self.required_permissions),
            "version": self.version,
        }


class BaseTool(ABC):
    """
    Abstract base class for all tools.

    Tools must implement the execute() method and provide a definition.
    """

    def __init__(self):
        self._execution_count = 0
        self._total_execution_time = 0.0
        self._error_count = 0

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the tool's definition"""
        pass

    @abstractmethod
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        """
        Execute the tool with given parameters.

        Args:
            parameters: Tool parameters as key-value pairs
            context: Execution context (tenant_id, permissions, etc.)

        Returns:
            ToolResult with output or error
        """
        pass

    def validate_parameters(self, parameters: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate input parameters against definition"""
        errors = []

        for param_def in self.definition.parameters:
            value = parameters.get(param_def.name, param_def.default)
            valid, error = param_def.validate(value)
            if not valid:
                errors.append(error)

        # Check for unknown parameters
        known_params = {p.name for p in self.definition.parameters}
        for key in parameters:
            if key not in known_params:
                errors.append(f"Unknown parameter: {key}")

        return len(errors) == 0, errors

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return {
            "execution_count": self._execution_count,
            "total_execution_time_ms": self._total_execution_time,
            "avg_execution_time_ms": (
                self._total_execution_time / self._execution_count
                if self._execution_count > 0 else 0
            ),
            "error_count": self._error_count,
            "error_rate": (
                self._error_count / self._execution_count
                if self._execution_count > 0 else 0
            ),
        }


@dataclass
class TenantPolicy:
    """Security policy for a tenant"""
    tenant_id: str
    allowed_tools: Optional[Set[str]] = None  # None = all allowed
    blocked_tools: Set[str] = field(default_factory=set)
    allowed_categories: Optional[Set[ToolCategory]] = None
    blocked_categories: Set[ToolCategory] = field(default_factory=set)
    max_security_level: ToolSecurityLevel = ToolSecurityLevel.STANDARD
    permissions: Set[str] = field(default_factory=set)

    # Limits
    max_concurrent_executions: int = 10
    max_execution_time_seconds: float = 300.0

    # Logging
    log_all_executions: bool = True
    log_parameters: bool = True
    log_outputs: bool = True


class ToolRegistry:
    """
    Centralized registry of available tools.

    Provides:
    - Tool registration and discovery
    - Permission and security checks
    - Tenant-scoped availability
    - Execution with sandboxing
    - Metrics and logging

    Usage:
        registry = ToolRegistry()

        # Register a tool
        registry.register(ShellTool())

        # Get tool by name
        tool = registry.get("shell")

        # Execute with security checks
        result = await registry.execute(
            "shell",
            {"command": "ls -la"},
            context={"tenant_id": "tenant_123"}
        )
    """

    def __init__(
        self,
        security_scanner=None,
        brain_logger=None,
        default_policy: Optional[TenantPolicy] = None,
    ):
        """
        Initialize the tool registry.

        Args:
            security_scanner: SecurityScanner for input/output scanning
            brain_logger: BrainLogger for audit logging
            default_policy: Default tenant policy for unregistered tenants
        """
        self._tools: Dict[str, BaseTool] = {}
        self._tenant_policies: Dict[str, TenantPolicy] = {}
        self._security_scanner = security_scanner
        self._brain_logger = brain_logger

        self._default_policy = default_policy or TenantPolicy(
            tenant_id="default",
            max_security_level=ToolSecurityLevel.STANDARD,
        )

        # Execution tracking
        self._active_executions: Dict[str, Dict[str, Any]] = {}
        self._execution_history: List[Dict[str, Any]] = []

        # Callbacks for execution events
        self._pre_execution_hooks: List[Callable] = []
        self._post_execution_hooks: List[Callable] = []

        logger.info("ToolRegistry initialized")

    def register(self, tool: BaseTool) -> bool:
        """
        Register a tool with the registry.

        Args:
            tool: Tool instance to register

        Returns:
            True if registered successfully
        """
        name = tool.definition.name

        if name in self._tools:
            logger.warning(f"Tool '{name}' already registered, replacing")

        self._tools[name] = tool
        logger.info(f"Registered tool: {name} ({tool.definition.category.value})")
        return True

    def unregister(self, tool_name: str) -> bool:
        """Unregister a tool"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info(f"Unregistered tool: {tool_name}")
            return True
        return False

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self._tools.get(tool_name)

    def list_tools(
        self,
        category: Optional[ToolCategory] = None,
        security_level: Optional[ToolSecurityLevel] = None,
        tenant_id: Optional[str] = None,
    ) -> List[ToolDefinition]:
        """
        List available tools with optional filtering.

        Args:
            category: Filter by category
            security_level: Filter by max security level
            tenant_id: Filter by tenant availability

        Returns:
            List of tool definitions
        """
        tools = []

        for name, tool in self._tools.items():
            defn = tool.definition

            # Apply filters
            if category and defn.category != category:
                continue

            if security_level:
                level_order = list(ToolSecurityLevel)
                if level_order.index(defn.security_level) > level_order.index(security_level):
                    continue

            if tenant_id:
                if not self._check_tenant_access(name, tenant_id):
                    continue

            tools.append(defn)

        return tools

    def set_tenant_policy(self, policy: TenantPolicy):
        """Set security policy for a tenant"""
        self._tenant_policies[policy.tenant_id] = policy
        logger.info(f"Set policy for tenant: {policy.tenant_id}")

    def get_tenant_policy(self, tenant_id: str) -> TenantPolicy:
        """Get policy for a tenant (returns default if not set)"""
        return self._tenant_policies.get(tenant_id, self._default_policy)

    def _check_tenant_access(self, tool_name: str, tenant_id: str) -> bool:
        """Check if a tenant can access a tool"""
        tool = self._tools.get(tool_name)
        if not tool:
            return False

        defn = tool.definition
        policy = self.get_tenant_policy(tenant_id)

        # Check if tool is explicitly blocked
        if tool_name in policy.blocked_tools:
            return False

        # Check if tool's category is blocked
        if defn.category in policy.blocked_categories:
            return False

        # Check if allowed list is set and tool is not in it
        if policy.allowed_tools is not None and tool_name not in policy.allowed_tools:
            return False

        # Check if allowed categories is set
        if policy.allowed_categories is not None and defn.category not in policy.allowed_categories:
            return False

        # Check security level
        level_order = list(ToolSecurityLevel)
        if level_order.index(defn.security_level) > level_order.index(policy.max_security_level):
            return False

        # Check if tool is blocked for this tenant in the tool definition
        if tenant_id in defn.blocked_tenants:
            return False

        # Check required permissions
        if defn.required_permissions:
            if not defn.required_permissions.issubset(policy.permissions):
                return False

        return True

    def add_pre_execution_hook(self, hook: Callable):
        """Add a hook to run before tool execution"""
        self._pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: Callable):
        """Add a hook to run after tool execution"""
        self._post_execution_hooks.append(hook)

    async def execute(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """
        Execute a tool with security checks and logging.

        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            context: Execution context (tenant_id, permissions, etc.)
            timeout: Override timeout in seconds

        Returns:
            ToolResult with output or error
        """
        context = context or {}
        tenant_id = context.get("tenant_id")
        execution_id = str(uuid.uuid4())
        start_time = time.time()
        security_warnings = []

        # Get the tool
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Tool '{tool_name}' not found",
                error_type="ToolNotFoundError",
            )

        defn = tool.definition

        # Check tenant access
        if tenant_id:
            if not self._check_tenant_access(tool_name, tenant_id):
                return ToolResult(
                    tool_name=tool_name,
                    status=ToolExecutionStatus.BLOCKED,
                    error=f"Tenant '{tenant_id}' does not have access to tool '{tool_name}'",
                    error_type="AccessDeniedError",
                )

        # Validate parameters
        valid, errors = tool.validate_parameters(parameters)
        if not valid:
            return ToolResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                error=f"Parameter validation failed: {errors}",
                error_type="ValidationError",
            )

        # Security scan inputs
        if self._security_scanner:
            for key, value in parameters.items():
                if isinstance(value, str):
                    scan_result = self._security_scanner.scan(value)
                    if scan_result.has_secrets:
                        security_warnings.append(f"Secrets detected in parameter '{key}'")
                        if defn.security_level != ToolSecurityLevel.DANGEROUS:
                            return ToolResult(
                                tool_name=tool_name,
                                status=ToolExecutionStatus.BLOCKED,
                                error="Secrets detected in parameters",
                                error_type="SecurityError",
                                security_warnings=security_warnings,
                            )

        # Check approval requirement
        if defn.requires_approval and not context.get("approved"):
            return ToolResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.BLOCKED,
                error="Tool requires explicit approval",
                error_type="ApprovalRequiredError",
            )

        # Run pre-execution hooks
        for hook in self._pre_execution_hooks:
            try:
                hook_result = hook(tool_name, parameters, context)
                if hook_result is False:
                    return ToolResult(
                        tool_name=tool_name,
                        status=ToolExecutionStatus.BLOCKED,
                        error="Blocked by pre-execution hook",
                        error_type="HookBlockedError",
                    )
            except Exception as e:
                logger.warning(f"Pre-execution hook failed: {e}")

        # Track active execution
        self._active_executions[execution_id] = {
            "tool_name": tool_name,
            "started_at": datetime.now(),
            "tenant_id": tenant_id,
        }

        # Execute the tool
        result = None
        effective_timeout = timeout or defn.timeout_seconds

        try:
            if defn.is_async:
                result = await asyncio.wait_for(
                    tool.execute(parameters, context),
                    timeout=effective_timeout
                )
            else:
                # Run sync tool in executor
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        lambda: asyncio.run(tool.execute(parameters, context))
                    ),
                    timeout=effective_timeout
                )

            # Update stats
            tool._execution_count += 1
            execution_time = (time.time() - start_time) * 1000
            tool._total_execution_time += execution_time

            if result.status == ToolExecutionStatus.FAILED:
                tool._error_count += 1

            result.execution_time_ms = execution_time
            result.completed_at = datetime.now()
            result.security_warnings = security_warnings

        except asyncio.TimeoutError:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.TIMEOUT,
                error=f"Tool execution timed out after {effective_timeout}s",
                error_type="TimeoutError",
                execution_time_ms=(time.time() - start_time) * 1000,
                completed_at=datetime.now(),
                security_warnings=security_warnings,
            )
            tool._error_count += 1

        except Exception as e:
            result = ToolResult(
                tool_name=tool_name,
                status=ToolExecutionStatus.FAILED,
                error=str(e),
                error_type=type(e).__name__,
                execution_time_ms=(time.time() - start_time) * 1000,
                completed_at=datetime.now(),
                security_warnings=security_warnings,
            )
            tool._error_count += 1
            logger.error(f"Tool execution failed: {e}")

        finally:
            # Remove from active executions
            self._active_executions.pop(execution_id, None)

        # Security scan output
        if self._security_scanner and result.output and isinstance(result.output, str):
            scan_result = self._security_scanner.scan(result.output)
            if scan_result.has_pii:
                result.security_warnings.append("PII detected in output")
            if scan_result.has_secrets:
                result.security_warnings.append("Secrets detected in output")

        # Run post-execution hooks
        for hook in self._post_execution_hooks:
            try:
                hook(tool_name, parameters, context, result)
            except Exception as e:
                logger.warning(f"Post-execution hook failed: {e}")

        # Log to history
        self._execution_history.append({
            "execution_id": execution_id,
            "tool_name": tool_name,
            "tenant_id": tenant_id,
            "status": result.status.value,
            "execution_time_ms": result.execution_time_ms,
            "timestamp": result.started_at.isoformat(),
        })

        # Limit history size
        if len(self._execution_history) > 1000:
            self._execution_history = self._execution_history[-500:]

        return result

    def execute_sync(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> ToolResult:
        """Synchronous wrapper for execute"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.execute(tool_name, parameters, context, timeout)
        )

    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get execution statistics for a tool"""
        tool = self._tools.get(tool_name)
        if tool:
            return tool.get_stats()
        return None

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get overall registry statistics"""
        return {
            "registered_tools": len(self._tools),
            "tools_by_category": {
                cat.value: sum(
                    1 for t in self._tools.values()
                    if t.definition.category == cat
                )
                for cat in ToolCategory
            },
            "active_executions": len(self._active_executions),
            "recent_executions": len(self._execution_history),
            "tenant_policies": len(self._tenant_policies),
        }

    def get_recent_executions(
        self,
        limit: int = 50,
        tool_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get recent execution history"""
        history = self._execution_history.copy()

        if tool_name:
            history = [h for h in history if h["tool_name"] == tool_name]

        if tenant_id:
            history = [h for h in history if h.get("tenant_id") == tenant_id]

        return history[-limit:]


# Global registry instance
_default_registry: Optional[ToolRegistry] = None


def get_tool_registry() -> ToolRegistry:
    """Get or create the default ToolRegistry instance"""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def set_tool_registry(registry: ToolRegistry):
    """Set the default ToolRegistry instance"""
    global _default_registry
    _default_registry = registry


def register_tool(tool: BaseTool):
    """Convenience function to register a tool with the default registry"""
    return get_tool_registry().register(tool)


def tool(
    name: str,
    description: str,
    category: ToolCategory = ToolCategory.UTILITY,
    security_level: ToolSecurityLevel = ToolSecurityLevel.SAFE,
    parameters: Optional[List[ToolParameter]] = None,
    timeout_seconds: float = 60.0,
    is_async: bool = False,
):
    """
    Decorator to create a tool from a function.

    Usage:
        @tool(
            name="add_numbers",
            description="Add two numbers together",
            category=ToolCategory.COMPUTE,
            parameters=[
                ToolParameter("a", "float", "First number"),
                ToolParameter("b", "float", "Second number"),
            ]
        )
        async def add_numbers(a: float, b: float, context: dict = None) -> ToolResult:
            return ToolResult(
                tool_name="add_numbers",
                status=ToolExecutionStatus.SUCCESS,
                output=a + b
            )
    """
    def decorator(func: Callable):
        # Create a tool class from the function
        class FunctionTool(BaseTool):
            def __init__(self):
                super().__init__()
                self._func = func
                self._definition = ToolDefinition(
                    name=name,
                    description=description,
                    category=category,
                    security_level=security_level,
                    parameters=parameters or [],
                    timeout_seconds=timeout_seconds,
                    is_async=is_async or asyncio.iscoroutinefunction(func),
                )

            @property
            def definition(self) -> ToolDefinition:
                return self._definition

            async def execute(
                self,
                parameters: Dict[str, Any],
                context: Optional[Dict[str, Any]] = None,
            ) -> ToolResult:
                try:
                    if asyncio.iscoroutinefunction(self._func):
                        result = await self._func(**parameters, context=context)
                    else:
                        result = self._func(**parameters, context=context)

                    # If function returns ToolResult, use it
                    if isinstance(result, ToolResult):
                        return result

                    # Otherwise wrap the output
                    return ToolResult(
                        tool_name=name,
                        status=ToolExecutionStatus.SUCCESS,
                        output=result,
                    )

                except Exception as e:
                    return ToolResult(
                        tool_name=name,
                        status=ToolExecutionStatus.FAILED,
                        error=str(e),
                        error_type=type(e).__name__,
                    )

        # Create and register the tool
        tool_instance = FunctionTool()
        register_tool(tool_instance)

        # Return the original function for direct use
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._tool = tool_instance
        return wrapper

    return decorator
