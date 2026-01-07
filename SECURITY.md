# Security Best Practices

## Dependency Management

### Pinned Dependencies for Docker Builds

To mitigate supply-chain attacks and ensure reproducible builds, this project uses pinned dependency versions for Docker image builds.

**Files:**
- `requirements.txt` - Development requirements with version ranges for flexibility
- `requirements.lock` - Production requirements with exact pinned versions for Docker builds

### Why Pinned Dependencies?

Unpinned dependencies in Docker builds create security risks:

1. **Supply Chain Attacks**: A compromised package release on PyPI could be automatically pulled into your build
2. **Reproducibility**: Builds may produce different results at different times
3. **Dependency Confusion**: Version ranges can lead to unexpected dependency resolution

### How It Works

The `Dockerfile` uses `requirements.lock` instead of `requirements.txt`:

```dockerfile
# Install python dependencies
# Use pinned requirements for reproducible builds and supply-chain security
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip==25.0.0 && \
    pip install --no-cache-dir -r requirements.lock
```

### Updating Dependencies

When adding or updating dependencies:

1. Update `requirements.txt` with the new package or version range
2. Test the changes in your development environment
3. Update `requirements.lock` with the exact pinned version you've tested
4. Commit both files to version control

#### Regenerating requirements.lock

You can use `pip-tools` to generate a new lock file:

```bash
# Install pip-tools
pip install pip-tools

# Generate pinned requirements
pip-compile --generate-hashes --allow-unsafe --output-file=requirements.lock requirements.txt
```

**Note**: Hash verification (`--generate-hashes`) provides additional security by verifying package integrity, but requires more maintenance when updating dependencies.

### Additional Security Measures

1. **Non-root user**: The Docker container runs as a non-root user (`appuser`)
2. **No cache**: `--no-cache-dir` flag prevents caching of pip packages
3. **Pinned pip version**: Even pip itself is pinned to a specific version
4. **Minimal base image**: Uses `python:3.10-slim` to reduce attack surface

## Reporting Security Issues

If you discover a security vulnerability, please email the maintainers directly rather than opening a public issue.
