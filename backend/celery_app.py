from celery import Celery
import os

REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
celery_app = Celery("transit", broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.task_time_limit = 600
celery_app.conf.worker_concurrency = 2
