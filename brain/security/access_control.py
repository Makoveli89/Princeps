"""Access Control - Permission management."""
import functools
from dataclasses import dataclass, field
from enum import Enum


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    INGEST = "ingest"
    DISTILL = "distill"
    QUERY = "query"

@dataclass
class Role:
    name: str
    permissions: set[Permission] = field(default_factory=set)

    def has_permission(self, permission: Permission) -> bool:
        return permission in self.permissions or Permission.ADMIN in self.permissions

# Default roles
ROLES = {
    "admin": Role("admin", {Permission.ADMIN}),
    "user": Role("user", {Permission.READ, Permission.QUERY}),
    "editor": Role("editor", {Permission.READ, Permission.WRITE, Permission.QUERY, Permission.INGEST}),
}

class AccessControl:
    def __init__(self, session=None):
        self.session = session

    def check(self, user_id: str, permission: Permission, resource_id: str | None = None) -> bool:
        # Stub implementation - always returns True
        return True

    def get_user_roles(self, user_id: str) -> list[Role]:
        return [ROLES["user"]]

def check_permission(user_id: str, permission: Permission) -> bool:
    return AccessControl().check(user_id, permission)

def require_permission(permission: Permission):
    """Decorator to require a permission."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Stub - in production, check user from context
            return func(*args, **kwargs)
        return wrapper
    return decorator
