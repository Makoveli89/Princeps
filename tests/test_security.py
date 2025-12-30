"""Tests for security features - PII detection, tenant isolation, logging safety."""
import pytest
from uuid import uuid4

from brain.security.security_scanner import (
    SecurityScanner,
    PIIPattern,
    SecretPattern,
    ScanResult,
    scan_content,
    redact_pii,
)
from brain.security.tenant_isolation import (
    TenantContext,
    TenantIsolation,
    ensure_tenant_access,
    get_tenant_filter,
)
from brain.core.models import Document, DocChunk, Operation


class TestPIIDetection:
    """Tests for PII detection patterns."""

    def test_email_detection(self):
        """Should detect various email formats."""
        scanner = SecurityScanner()

        test_cases = [
            "user@example.com",
            "user.name@subdomain.example.org",
            "user+tag@example.co.uk",
            "USER@EXAMPLE.COM",
        ]

        for email in test_cases:
            result = scanner.scan(f"Contact: {email}")
            assert result.has_pii, f"Failed to detect email: {email}"
            assert "email" in result.pii_found

    def test_phone_detection(self):
        """Should detect various phone number formats."""
        scanner = SecurityScanner()

        test_cases = [
            "555-123-4567",
            "555.123.4567",
            "5551234567",
        ]

        for phone in test_cases:
            result = scanner.scan(f"Call: {phone}")
            assert result.has_pii, f"Failed to detect phone: {phone}"
            assert "phone" in result.pii_found

    def test_ssn_detection(self):
        """Should detect SSN patterns."""
        scanner = SecurityScanner()

        test_cases = [
            "123-45-6789",
            "123456789",
        ]

        for ssn in test_cases:
            result = scanner.scan(f"SSN: {ssn}")
            assert result.has_pii, f"Failed to detect SSN: {ssn}"
            assert "ssn" in result.pii_found


class TestSecretDetection:
    """Tests for secret/credential detection."""

    def test_api_key_detection(self):
        """Should detect API key patterns."""
        scanner = SecurityScanner()

        test_cases = [
            "api_key=abcdefghijklmnopqrstuvwxyz",
            "apikey: abcdefghijklmnopqrstuvwxyz123456",
            'API_KEY="sk_live_abcdefghijklmnopqrstuvwxyz12345678"',
        ]

        for key in test_cases:
            result = scanner.scan(key)
            assert result.has_secrets, f"Failed to detect API key in: {key}"
            assert "api_key" in result.secrets_found

    def test_password_detection(self):
        """Should detect password patterns."""
        scanner = SecurityScanner()

        test_cases = [
            "password=MySecretPassword123!",
            "passwd: secretvalue!",
            'pwd="complex_password_here"',
        ]

        for pwd in test_cases:
            result = scanner.scan(pwd)
            assert result.has_secrets, f"Failed to detect password in: {pwd}"
            assert "password" in result.secrets_found

    def test_github_token_detection(self):
        """Should detect GitHub personal access tokens."""
        scanner = SecurityScanner()
        token = "ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"  # 40 chars after ghp_
        result = scanner.scan(f"GITHUB_TOKEN={token}")
        assert result.has_secrets
        assert "github_token" in result.secrets_found

    def test_openai_key_detection(self):
        """Should detect OpenAI API keys."""
        scanner = SecurityScanner()
        key = "sk-" + "x" * 48  # OpenAI key pattern
        result = scanner.scan(f"OPENAI_API_KEY={key}")
        assert result.has_secrets
        assert "openai_key" in result.secrets_found


class TestScanResultFlags:
    """Tests to verify correct flags are set in ScanResult."""

    def test_has_pii_flag(self):
        """has_pii should be True only when PII is found."""
        result_with_pii = scan_content("Email: test@example.com")
        result_without_pii = scan_content("No sensitive data here")

        assert result_with_pii.has_pii is True
        assert result_without_pii.has_pii is False

    def test_has_secrets_flag(self):
        """has_secrets should be True only when secrets are found."""
        result_with_secret = scan_content("api_key=secretvalue12345678901234")
        result_without_secret = scan_content("No sensitive data here")

        assert result_with_secret.has_secrets is True
        assert result_without_secret.has_secrets is False

    def test_pii_found_list(self):
        """pii_found should contain all detected PII types."""
        content = "Email: test@example.com, Phone: 555-123-4567"
        result = scan_content(content)

        assert "email" in result.pii_found
        assert "phone" in result.pii_found

    def test_secrets_found_list(self):
        """secrets_found should contain all detected secret types."""
        content = "api_key=abc123456789012345678901234 password=mysecretpassword"
        result = scan_content(content)

        assert "api_key" in result.secrets_found
        assert "password" in result.secrets_found


class TestPIIRedaction:
    """Tests for PII redaction functionality."""

    def test_redact_email(self):
        """Should redact email addresses."""
        content = "Contact: user@example.com for info"
        redacted = redact_pii(content)
        assert "user@example.com" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_phone(self):
        """Should redact phone numbers."""
        content = "Call 555-123-4567 now!"
        redacted = redact_pii(content)
        assert "555-123-4567" not in redacted
        assert "[REDACTED]" in redacted

    def test_redact_multiple(self):
        """Should redact multiple sensitive items."""
        content = "Email: a@b.com, Phone: 555-555-5555, SSN: 123-45-6789"
        redacted = redact_pii(content)

        assert "a@b.com" not in redacted
        assert "555-555-5555" not in redacted
        assert "123-45-6789" not in redacted
        assert redacted.count("[REDACTED]") >= 3


class TestTenantIsolation:
    """Tests for tenant isolation context."""

    def test_tenant_context_set_and_get(self):
        """TenantContext should properly set and get tenant ID."""
        TenantContext.clear()
        assert TenantContext.get() is None

        TenantContext.set("tenant-123")
        assert TenantContext.get() == "tenant-123"

        TenantContext.clear()
        assert TenantContext.get() is None

    def test_tenant_isolation_context_manager(self):
        """TenantIsolation should properly scope tenant ID."""
        TenantContext.clear()

        with TenantIsolation("tenant-456"):
            assert TenantContext.get() == "tenant-456"

        assert TenantContext.get() is None

    def test_ensure_tenant_access(self, session):
        """ensure_tenant_access should verify tenant ownership."""
        tenant_id = str(uuid4())
        other_tenant_id = str(uuid4())

        # Same tenant should have access
        assert ensure_tenant_access(session, tenant_id, tenant_id) is True

        # Different tenant should not have access
        assert ensure_tenant_access(session, tenant_id, other_tenant_id) is False


class TestTenantFiltering:
    """Tests for tenant filtering in database queries."""

    def test_get_tenant_filter_for_document(self):
        """Should generate correct filter for Document model."""
        tenant_id = uuid4()
        filter_clause = get_tenant_filter(Document, tenant_id)
        # The filter should be a SQLAlchemy BinaryExpression
        assert filter_clause is not None

    def test_get_tenant_filter_for_chunk(self):
        """Should generate correct filter for DocChunk model."""
        tenant_id = uuid4()
        filter_clause = get_tenant_filter(DocChunk, tenant_id)
        assert filter_clause is not None

    def test_get_tenant_filter_for_operation(self):
        """Should generate correct filter for Operation model."""
        tenant_id = uuid4()
        filter_clause = get_tenant_filter(Operation, tenant_id)
        assert filter_clause is not None


class TestCustomPatterns:
    """Tests for custom PII/secret patterns."""

    def test_custom_pii_pattern(self):
        """Should support custom PII patterns."""
        custom_pattern = PIIPattern(
            name="credit_card",
            pattern=r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            severity="high"
        )
        scanner = SecurityScanner(pii_patterns=[custom_pattern])

        result = scanner.scan("CC: 4111-1111-1111-1111")
        assert result.has_pii
        assert "credit_card" in result.pii_found

    def test_custom_secret_pattern(self):
        """Should support custom secret patterns."""
        custom_pattern = SecretPattern(
            name="aws_key",
            pattern=r'AKIA[0-9A-Z]{16}',
            severity="critical"
        )
        scanner = SecurityScanner(secret_patterns=[custom_pattern])

        result = scanner.scan("AWS_KEY=AKIAIOSFODNN7EXAMPLE")
        assert result.has_secrets
        assert "aws_key" in result.secrets_found
