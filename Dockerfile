FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# libpq-dev and gcc are often needed for psycopg2 (if not binary) or other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose port
EXPOSE 80

# Environment variables
ENV MODULE_NAME=server
ENV VARIABLE_NAME=app
ENV PORT=80

# Run command
CMD ["gunicorn", "-c", "gunicorn_conf.py", "server:app"]
