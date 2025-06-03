#!/bin/bash

# Start Redis (in background)
redis-server &

# Start Celery worker (in background)
celery -A app.worker worker --loglevel=info &

# Start FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000
