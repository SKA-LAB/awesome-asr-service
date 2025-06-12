import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import json
from datetime import datetime

# Import the FastAPI app
from app.main import app

# Create test client
client = TestClient(app)

# Mock data and fixtures
@pytest.fixture
def mock_audio_file():
    """Create a mock audio file for testing."""
    file_path = "tests/fixtures/test_audio.mp3"
    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Create a dummy MP3 file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            # Write minimal MP3 header
            f.write(b"\xFF\xFB\x90\x44\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00")
    
    return file_path

@pytest.fixture
def mock_transcription():
    return "This is a test transcription of a meeting. John will follow up with the client next week."

@pytest.fixture
def mock_summary():
    return """## Meeting Summary
This was a test meeting where action items were discussed.

## Action Items
- John will follow up with the client next week
"""

@pytest.fixture
def mock_conversation_history():
    return [
        {"role": "user", "content": "What was discussed in the last meeting?"},
        {"role": "assistant", "content": "In the last meeting, the team discussed project timelines and client feedback."}
    ]

# Test health check endpoint
def test_health_check():
    with patch('app.main.ChatOllama') as mock_ollama, \
         patch('app.worker.app.control.ping') as mock_ping:
        
        # Configure mocks
        mock_ollama.return_value.invoke.return_value = "Hello"
        mock_ping.return_value = [{'worker1': {'ok': 'pong'}}]
        
        # Make request
        response = client.get("/health")
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == {
            "status": "healthy", 
            "services": {"ollama": "available", "redis": "available"}
        }

# Test transcribe endpoint
def test_transcribe(mock_audio_file):
    with patch('app.main.transcribe_audio_task') as mock_task:
        # Configure mock
        mock_task.delay.return_value = MagicMock(id="test-task-id")
        
        # Open test file
        with open(mock_audio_file, "rb") as f:
            # Make request
            response = client.post(
                "/transcribe/",
                files={"file": ("test_audio.mp3", f, "audio/mpeg")}
            )
        
        # Assertions
        assert response.status_code == 200
        assert "task_id" in response.json()
        assert "message" in response.json()
        mock_task.delay.assert_called_once()

# Test get_result endpoint
def test_get_result_processing():
    # Test when result is not ready
    with patch('os.path.exists') as mock_exists:
        mock_exists.return_value = False
        
        response = client.get("/result/test-task-id")
        
        assert response.status_code == 202
        assert response.json() == {"status": "processing"}

def test_get_result_ready(mock_transcription):
    # Test when result is ready
    with patch('os.path.exists') as mock_exists, \
         patch('builtins.open') as mock_open, \
         patch('os.remove') as mock_remove:
        
        mock_exists.return_value = True
        mock_file = MagicMock()
        mock_file.__enter__.return_value.read.return_value = mock_transcription
        mock_open.return_value = mock_file
        
        response = client.get("/result/test-task-id")
        
        assert response.status_code == 200
        assert response.json() == {"transcription": mock_transcription}
        mock_remove.assert_called_once()

# Test summarize endpoint
def test_summarize(mock_transcription, mock_summary):
    with patch('app.main.ChatOllama') as mock_ollama:
        # Configure mock
        mock_response = MagicMock()
        mock_response.content = mock_summary
        mock_ollama.return_value.invoke.return_value = mock_response
        
        # Make request
        response = client.post(
            "/summarize/",
            json={"transcript": mock_transcription}
        )
        
        # Assertions
        assert response.status_code == 200
        assert response.json() == {"summary_markdown": mock_summary}

# Test transcribe_and_summarize endpoint
def test_transcribe_and_summarize(mock_audio_file, mock_transcription, mock_summary):
    with patch('app.main.save_uploaded_file') as mock_save, \
         patch('app.main.process_audio_file') as mock_process, \
         patch('app.main.save_to_file_system') as mock_save_file, \
         patch('app.main.save_to_notion_db') as mock_save_notion:
        
        # Configure mocks
        mock_process.return_value = ("Test Meeting", mock_transcription, mock_summary)
        mock_save_file.return_value = "/meeting-notes/test_meeting.md"
        mock_save_notion.return_value = "https://notion.so/test-page"
        
        # Open test file
        with open(mock_audio_file, "rb") as f:
            # Make request
            response = client.post(
                "/transcribe-and-summarize/",
                files={"file": ("test_audio.mp3", f, "audio/mpeg")},
                data={
                    "save_to_file": "true",
                    "save_to_notion": "true",
                    "meeting_date": "2023-05-15",
                    "participants": json.dumps(["John", "Jane"])
                }
            )
        
        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["title"] == "Test Meeting"
        assert result["transcription"] == mock_transcription
        assert result["summary_markdown"] == mock_summary
        assert result["saved_to_file"] == "/meeting-notes/test_meeting.md"
        assert result["saved_to_notion"] == "https://notion.so/test-page"

# Test semantic search endpoints
def test_get_indexed_files():
    with patch('app.main.get_search_engine') as mock_get_engine:
        # Configure mock
        mock_engine = MagicMock()
        mock_engine.metadata = {
            "file1.md": {
                "display_name": "Meeting 1",
                "last_modified": datetime.now().timestamp(),
                "summary_idx_active": True,
                "transcript_indices_active": [True, True]
            }
        }
        mock_engine.summary_index.ntotal = 1
        mock_engine.transcript_index.ntotal = 10
        mock_get_engine.return_value = mock_engine
        
        # Make request
        response = client.get("/semantic-search/indexed-files")
        
        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert len(result["indexed_files"]) == 1
        assert result["summary_count"] == 1
        assert result["transcript_chunk_count"] == 10

def test_rebuild_search_index():
    with patch('app.main.load_meeting_files') as mock_load_files, \
         patch('app.main.get_search_engine') as mock_get_engine:
        
        # Configure mocks
        mock_load_files.return_value = ["file1.md", "file2.md"]
        mock_engine = MagicMock()
        mock_engine.summary_index.ntotal = 2
        mock_engine.transcript_index.ntotal = 20
        mock_get_engine.return_value = mock_engine
        
        # Make request
        response = client.post("/semantic-search/rebuild-index")
        
        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["indexed_files_count"] == 2
        assert result["summary_count"] == 2
        assert result["transcript_chunk_count"] == 20
        mock_engine.reindex_all.assert_called_once_with(["file1.md", "file2.md"])

def test_semantic_search():
    with patch('app.indexing_utils.perform_semantic_search') as mock_search:
        # Configure mock
        mock_search.return_value = [
            {
                "file_id": "file1.md",
                "display_name": "Meeting 1",
                "score": 0.85,
                "content": "This is a matching content",
                "type": "summary"
            }
        ]
        
        # Make request
        response = client.get(
            "/semantic-search/search?query=test&search_in_summaries=true&search_in_transcripts=true&top_k=5"
        )
        
        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert result["query"] == "test"
        assert len(result["results"]) == 1
        assert result["results"][0]["file_id"] == "file1.md"

# Test RAG search endpoint
def test_rag_search(mock_conversation_history):
    with patch('app.main.get_rag_search_engine') as mock_get_engine:
        # Configure mock
        mock_engine = MagicMock()
        mock_engine.search_with_history.return_value = {
            "success": True,
            "answer": "Based on the meeting notes, John will follow up with the client next week.",
            "context_used": ["This is a test transcription of a meeting. John will follow up with the client next week."]
        }
        mock_get_engine.return_value = mock_engine
        
        # Make request
        response = client.post(
            "/rag-search",
            json={
                "query": "What action items were assigned?",
                "conversation_history": mock_conversation_history,
                "limit_files": ["file1.md"]
            }
        )
        
        # Assertions
        assert response.status_code == 200
        result = response.json()
        assert result["success"] is True
        assert "answer" in result
        assert "context_used" in result
        
        # Verify the engine was called with correct parameters
        mock_get_engine.assert_called_once_with(["file1.md"])
        mock_engine.search_with_history.assert_called_once()

# Test error handling
def test_empty_query_rag_search():
    response = client.post(
        "/rag-search",
        json={"query": ""}
    )
    
    assert response.status_code == 400
    assert response.json()["success"] is False
    assert "error" in response.json()

def test_invalid_file_type():
    response = client.post(
        "/transcribe/",
        files={"file": ("test.txt", b"test content", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "error" in response.json()
