import pytest
import os
import sys
from unittest.mock import patch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock environment variables for testing
@pytest.fixture(autouse=True)
def mock_env_vars():
    with patch.dict(os.environ, {
        "TEMP_DIR": "/tmp",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "OUTPUT_DIR": "/meeting-notes"
    }):
        yield

# Mock Redis and Celery for testing
@pytest.fixture(autouse=True)
def mock_celery():
    with patch('app.worker.app') as mock_app:
        mock_app.control.ping.return_value = [{'worker1': {'ok': 'pong'}}]
        yield mock_app
