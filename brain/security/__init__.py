"""
Brain Security Module
=====================

PII scanning, tenant isolation, and access control.

Exports:
    - SecurityScanner: PII and secrets detection
    - TenantIsolation: Multi-tenant data isolation
    - AccessControl: Permission management
    - Security utilities
"""

from .access_control import (
    AccessControl,
    Permission,
    Role,
    check_permission,
    require_permission,
)
from .security_scanner import (
    PIIPattern,
    ScanResult,
    SecretPattern,
    SecurityScanner,
    redact_pii,
    scan_content,
)
from .security_utils import (
    decrypt_field,
    encrypt_field,
    generate_api_key,
    hash_sensitive,
    validate_api_key,
)
from .tenant_isolation import (
    TenantContext,
    TenantIsolation,
    ensure_tenant_access,
    get_tenant_filter,
)

__all__ = [
    # Security Scanner
    "SecurityScanner",
    "ScanResult",
    "PIIPattern",
    "SecretPattern",
    "scan_content",
    "redact_pii",
    # Tenant Isolation
    "TenantIsolation",
    "TenantContext",
    "ensure_tenant_access",
    "get_tenant_filter",
    # Access Control
    "AccessControl",
    "Permission",
    "Role",
    "check_permission",
    "require_permission",
    # Utilities
    "hash_sensitive",
    "encrypt_field",
    "decrypt_field",
    "generate_api_key",
    "validate_api_key",
]
