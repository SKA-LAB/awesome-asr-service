#!/bin/bash
set -e

# Start Redis in background if not using external Redis
if [ "${USE_EXTERNAL_REDIS:-false}" = "false" ]; then
    echo "Starting Redis server..."
    redis-server --daemonize yes
    # Wait for Redis to be ready
    sleep 2
fi

# Start Celery worker in background
echo "Starting Celery worker..."
celery -A app.worker worker --loglevel=info --concurrency=2 &

# Wait for Celery to initialize
sleep 3

# Set API URL for the dashboard to connect to the FastAPI service
export API_URL="http://localhost:${PORT:-8000}"

# Start Streamlit dashboard in background
echo "Starting Streamlit dashboard..."
STREAMLIT_PORT=${DASHBOARD_PORT:-8501}
streamlit run app/dashboard.py --server.port $STREAMLIT_PORT --server.address 0.0.0.0 &

# Wait for Streamlit to initialize
sleep 2
echo "Streamlit dashboard running on port $STREAMLIT_PORT"

# Start FastAPI application with Uvicorn
echo "Starting FastAPI application..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1} ${UVICORN_EXTRA_ARGS:-}

# This trap ensures we kill background processes when the container stops
trap 'kill $(jobs -p)' EXIT