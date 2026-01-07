"""Security Scanner - PII and secrets detection."""

import re
from dataclasses import dataclass, field


@dataclass
class PIIPattern:
    name: str
    pattern: str
    severity: str = "medium"


@dataclass
class SecretPattern:
    name: str
    pattern: str
    severity: str = "high"


@dataclass
class ScanResult:
    has_pii: bool = False
    has_secrets: bool = False
    pii_found: list[str] = field(default_factory=list)
    secrets_found: list[str] = field(default_factory=list)
    redacted_content: str | None = None


class SecurityScanner:
    """Scan content for PII and secrets."""

    PII_PATTERNS = [
        PIIPattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
        PIIPattern("phone", r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),
        PIIPattern("ssn", r"\b\d{3}[-]?\d{2}[-]?\d{4}\b"),
    ]

    SECRET_PATTERNS = [
        SecretPattern("api_key", r'(?:api[_-]?key|apikey)\s*[:=]\s*["\']?[\w-]{20,}'),
        SecretPattern("password", r'(?:password|passwd|pwd)\s*[:=]\s*["\']?[\w!@#$%^&*-]{8,}'),
        SecretPattern("github_token", r"ghp_[a-zA-Z0-9]{36}"),
        SecretPattern("openai_key", r"sk-[a-zA-Z0-9]{48}"),
    ]

    def __init__(
        self,
        pii_patterns: list[PIIPattern] | None = None,
        secret_patterns: list[SecretPattern] | None = None,
    ):
        self.pii_patterns = pii_patterns or self.PII_PATTERNS
        self.secret_patterns = secret_patterns or self.SECRET_PATTERNS

    def scan(self, content: str) -> ScanResult:
        result = ScanResult()

        for pattern in self.pii_patterns:
            if re.search(pattern.pattern, content, re.IGNORECASE):
                result.pii_found.append(pattern.name)
                result.has_pii = True

        for pattern in self.secret_patterns:
            if re.search(pattern.pattern, content, re.IGNORECASE):
                result.secrets_found.append(pattern.name)
                result.has_secrets = True

        return result

    def redact(self, content: str) -> str:
        redacted = content
        for pattern in self.pii_patterns + self.secret_patterns:
            redacted = re.sub(pattern.pattern, "[REDACTED]", redacted, flags=re.IGNORECASE)
        return redacted


def scan_content(content: str) -> ScanResult:
    return SecurityScanner().scan(content)


def redact_pii(content: str) -> str:
    return SecurityScanner().redact(content)
