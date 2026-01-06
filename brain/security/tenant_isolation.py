"""Tenant Isolation - Multi-tenant data isolation."""

from contextlib import contextmanager
from uuid import UUID


class TenantContext:
    """Context for tenant-scoped operations."""

    _current_tenant: str | None = None

    @classmethod
    def set(cls, tenant_id: str):
        cls._current_tenant = tenant_id

    @classmethod
    def get(cls) -> str | None:
        return cls._current_tenant

    @classmethod
    def clear(cls):
        cls._current_tenant = None


@contextmanager
def TenantIsolation(tenant_id: str):
    """Context manager for tenant isolation."""
    previous = TenantContext.get()
    TenantContext.set(tenant_id)
    try:
        yield
    finally:
        if previous:
            TenantContext.set(previous)
        else:
            TenantContext.clear()


def ensure_tenant_access(session, tenant_id: str | UUID, resource_tenant_id: str | UUID) -> bool:
    """Ensure the current tenant has access to a resource."""
    return str(tenant_id) == str(resource_tenant_id)


def get_tenant_filter(model, tenant_id: str | UUID):
    """Get SQLAlchemy filter for tenant isolation."""
    if hasattr(model, "tenant_id"):
        return model.tenant_id == tenant_id
    return True
