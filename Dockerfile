FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --target=/app/deps -r requirements.txt

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/deps /usr/local/lib/python3.11/site-packages
COPY . /app

EXPOSE 80

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "-c", "gunicorn_conf.py", "server:app"]
