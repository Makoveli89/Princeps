# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing pyc files to disc
# PYTHONUNBUFFERED: Prevents Python from buffering stdout and stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
# Use pinned requirements for reproducible builds and supply-chain security
COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip==25.0.0 && \
    pip install --no-cache-dir -r requirements.lock

# Copy project
COPY . .

# Create a non-root user and switch to it
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# Using uvicorn directly. In production, consider gunicorn with uvicorn workers or the specific tiangolo image.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
