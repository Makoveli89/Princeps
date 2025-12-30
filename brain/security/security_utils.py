"""Security Utilities - Encryption, hashing, API keys."""
import base64
import hashlib
import secrets


def hash_sensitive(value: str, salt: str = "") -> str:
    """Hash sensitive data with SHA-256."""
    return hashlib.sha256((value + salt).encode()).hexdigest()

def encrypt_field(value: str, key: str | None = None) -> str:
    """Encrypt a field value (stub - uses base64 for demo)."""
    # In production, use proper encryption like Fernet
    return base64.b64encode(value.encode()).decode()

def decrypt_field(encrypted: str, key: str | None = None) -> str:
    """Decrypt a field value (stub - uses base64 for demo)."""
    return base64.b64decode(encrypted.encode()).decode()

def generate_api_key(prefix: str = "pk") -> str:
    """Generate a secure API key."""
    random_bytes = secrets.token_hex(24)
    return f"{prefix}_{random_bytes}"

def validate_api_key(api_key: str, expected_prefix: str = "pk") -> bool:
    """Validate API key format."""
    if not api_key:
        return False
    parts = api_key.split("_", 1)
    if len(parts) != 2:
        return False
    prefix, token = parts
    return prefix == expected_prefix and len(token) == 48
