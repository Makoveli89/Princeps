import multiprocessing
import os

host = os.getenv("HOST", "0.0.0.0")
port = os.getenv("PORT", "80")
bind_env = os.getenv("BIND", None)
use_bind = bind_env if bind_env else f"{host}:{port}"

workers_per_core_str = os.getenv("WORKERS_PER_CORE", "1")
max_workers_str = os.getenv("MAX_WORKERS")
web_concurrency_str = os.getenv("WEB_CONCURRENCY", None)

host = use_bind
bind = use_bind
keepalive = 120
errorlog = "-"
accesslog = "-"

if web_concurrency_str:
    workers = int(web_concurrency_str)
    assert workers > 0
else:
    cores = multiprocessing.cpu_count()
    workers_per_core = float(workers_per_core_str)
    default_web_concurrency = workers_per_core * cores
    if max_workers_str:
        workers = min(int(default_web_concurrency), int(max_workers_str))
    else:
        workers = int(default_web_concurrency)
    workers = max(int(workers), 2)  # Ensure at least 2 workers

loglevel = os.getenv("LOG_LEVEL", "info")
worker_class = "uvicorn.workers.UvicornWorker"
