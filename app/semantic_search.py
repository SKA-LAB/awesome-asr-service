import os
import json
import numpy as np
import faiss
import logging
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional, Tuple
import time
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("semantic-search")

# Constants
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-minilm")
EMBEDDING_DIMENSION = 384  # all-minilm dimension
FAISS_INDEX_DIR = os.environ.get("FAISS_INDEX_DIR", "/meeting-notes/.index")
METADATA_FILE = os.path.join(FAISS_INDEX_DIR, "metadata.json")
SUMMARY_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "summary_index.faiss")
TRANSCRIPT_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "transcript_index.faiss")

# Ensure index directory exists
os.makedirs(FAISS_INDEX_DIR, exist_ok=True)

class SemanticSearchEngine:
    def __init__(self):
        self.summary_index = None
        self.transcript_index = None
        self.metadata = {}
        self.load_or_create_indices()
    
    def load_or_create_indices(self):
        """Load existing indices or create new ones if they don't exist."""
        # Load metadata if it exists
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
        
        # Load or create summary index
        if os.path.exists(SUMMARY_INDEX_FILE):
            logger.info("Loading existing summary index")
            self.summary_index = faiss.read_index(SUMMARY_INDEX_FILE)
        else:
            logger.info("Creating new summary index")
            self.summary_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        
        # Load or create transcript index
        if os.path.exists(TRANSCRIPT_INDEX_FILE):
            logger.info("Loading existing transcript index")
            self.transcript_index = faiss.read_index(TRANSCRIPT_INDEX_FILE)
        else:
            logger.info("Creating new transcript index")
            self.transcript_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text using Ollama API."""
        # Retry mechanism for API calls
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/embed",
                    json={"model": EMBEDDING_MODEL, "input": text}
                )
                response.raise_for_status()
                embedding = np.array(response.json()["embeddings"], dtype=np.float32)
                return embedding
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to get embedding after {max_retries} attempts")
                    raise
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better embedding."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def index_document(self, file_info: Dict[str, Any], summary: str, transcript: str) -> None:
        """Index a document's summary and transcript."""
        file_id = file_info["path"]
        
        # Check if document is already indexed and unchanged
        if file_id in self.metadata:
            last_modified = os.path.getmtime(file_id)
            if last_modified <= self.metadata[file_id].get("last_modified", 0):
                logger.info(f"Document {file_id} already indexed and unchanged")
                return
        
        logger.info(f"Indexing document: {file_info['display_name']}")
        
        # Process summary
        if summary:
            logger.info(f"Embedding summary for {file_info['display_name']}")
            
            # Check if summary is long enough to warrant chunking
            if len(summary.split()) > 500:  # If summary has more than 500 words
                logger.info(f"Summary is long, chunking it for better embedding")
                summary_chunks = self.chunk_text(summary, chunk_size=500, overlap=100)
                
                # Remove old summary embeddings if they exist
                if file_id in self.metadata:
                    old_indices = self.metadata[file_id].get("summary_indices", [])
                    for idx in old_indices:
                        # Mark as inactive
                        self.metadata[file_id]["summary_indices_active"] = [False] * len(old_indices)
                
                # Add new embeddings
                summary_indices = []
                summary_chunks_text = []
                
                for i, chunk in enumerate(summary_chunks):
                    try:
                        chunk_embedding = self.get_embedding(chunk)
                        chunk_idx = self.summary_index.ntotal
                        self.summary_index.add(chunk_embedding)
                        summary_indices.append(chunk_idx)
                        summary_chunks_text.append(chunk)
                        logger.info(f"Added summary chunk {i+1}/{len(summary_chunks)} embedding at index {chunk_idx}")
                    except Exception as e:
                        logger.error(f"Error embedding summary chunk {i}: {str(e)}")
                
                # Update metadata
                if file_id not in self.metadata:
                    self.metadata[file_id] = {}
                
                self.metadata[file_id].update({
                    "summary_indices": summary_indices,
                    "summary_indices_active": [True] * len(summary_indices),
                    "summary_chunks": summary_chunks_text,
                    "is_chunked_summary": True,
                    "display_name": file_info["display_name"],
                    "date": file_info["date"].isoformat() if hasattr(file_info["date"], "isoformat") else str(file_info["date"]),
                    "last_modified": os.path.getmtime(file_id)
                })
            else:
                # For shorter summaries, keep as a single chunk
                summary_embedding = self.get_embedding(summary)
                logger.info(f"Finished embedding summary.")
                
                # Add to summary index
                if file_id in self.metadata:
                    # Remove old summary embedding
                    old_idx = self.metadata[file_id].get("summary_idx")
                    if old_idx is not None:
                        # We can't easily remove from FAISS, so we'll just track the active indices
                        self.metadata[file_id]["summary_idx_active"] = False
                
                # Add new embedding
                summary_idx = self.summary_index.ntotal
                self.summary_index.add(summary_embedding)
                logger.info(f"Added summary embedding at index {summary_idx}")
                
                # Update metadata
                if file_id not in self.metadata:
                    self.metadata[file_id] = {}
                
                self.metadata[file_id].update({
                    "summary_idx": summary_idx,
                    "summary_idx_active": True,
                    "is_chunked_summary": False,
                    "display_name": file_info["display_name"],
                    "date": file_info["date"].isoformat() if hasattr(file_info["date"], "isoformat") else str(file_info["date"]),
                    "last_modified": os.path.getmtime(file_id)
                })
        
        # Process transcript
        if transcript:
            # For transcript, we'll chunk it for better semantic search
            transcript_chunks = self.chunk_text(transcript)
            
            # Remove old transcript embeddings if they exist
            if file_id in self.metadata:
                old_indices = self.metadata[file_id].get("transcript_indices", [])
                for idx in old_indices:
                    # Mark as inactive
                    self.metadata[file_id]["transcript_indices_active"] = [False] * len(old_indices)
            
            # Add new embeddings
            transcript_indices = []
            transcript_chunks_text = []
            
            for i, chunk in enumerate(transcript_chunks):
                try:
                    chunk_embedding = self.get_embedding(chunk)
                    chunk_idx = self.transcript_index.ntotal
                    self.transcript_index.add(chunk_embedding)
                    transcript_indices.append(chunk_idx)
                    transcript_chunks_text.append(chunk)
                except Exception as e:
                    logger.error(f"Error embedding transcript chunk {i}: {str(e)}")
            logger.info(f"Added transcript embeddings at indices {transcript_indices}")
            
            # Update metadata
            if file_id not in self.metadata:
                self.metadata[file_id] = {}
            
            self.metadata[file_id].update({
                "transcript_indices": transcript_indices,
                "transcript_indices_active": [True] * len(transcript_indices),
                "transcript_chunks": transcript_chunks_text,
                "display_name": file_info["display_name"],
                "date": file_info["date"].isoformat() if hasattr(file_info["date"], "isoformat") else str(file_info["date"]),
                "last_modified": os.path.getmtime(file_id)
            })
        
        # Save indices and metadata
        self.save_indices()
    
    def save_indices(self) -> None:
        """Save indices and metadata to disk."""
        logger.info("Saving indices and metadata")
        faiss.write_index(self.summary_index, SUMMARY_INDEX_FILE)
        faiss.write_index(self.transcript_index, TRANSCRIPT_INDEX_FILE)
        
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f)
    
    def search(self, query: str, search_in_summaries: bool = True, 
               search_in_transcripts: bool = True, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for query in indexed documents."""
        if not query.strip():
            return []
        
        logger.info(f"Semantic search for: {query}")
        query_embedding = self.get_embedding(query)
        results = []
        
        # Search in summaries
        if search_in_summaries and self.summary_index.ntotal > 0:
            distances, indices = self.summary_index.search(query_embedding, top_k)
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    # Find document with this index
                    for file_id, meta in self.metadata.items():
                        # Check for single summary
                        if meta.get("is_chunked_summary", False) == False and meta.get("summary_idx") == idx and meta.get("summary_idx_active", True):
                            results.append({
                                "file_id": file_id,
                                "display_name": meta["display_name"],
                                "date": meta["date"],
                                "distance": float(distances[0][i]),
                                "type": "summary",
                                "context": None  # Will be populated later
                            })
                        
                        # Check for chunked summaries
                        elif meta.get("is_chunked_summary", False) == True:
                            summary_indices = meta.get("summary_indices", [])
                            summary_active = meta.get("summary_indices_active", [True] * len(summary_indices))
                            
                            for j, s_idx in enumerate(summary_indices):
                                if s_idx == idx and summary_active[j]:
                                    results.append({
                                        "file_id": file_id,
                                        "display_name": meta["display_name"],
                                        "date": meta["date"],
                                        "distance": float(distances[0][i]),
                                        "type": "summary_chunk",
                                        "chunk_index": j,
                                        "context": meta.get("summary_chunks", [])[j] if j < len(meta.get("summary_chunks", [])) else None
                                    })
        
        # Search in transcripts
        if search_in_transcripts and self.transcript_index.ntotal > 0:
            distances, indices = self.transcript_index.search(query_embedding, top_k)
            
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # Valid index
                    # Find document with this index
                    for file_id, meta in self.metadata.items():
                        transcript_indices = meta.get("transcript_indices", [])
                        transcript_active = meta.get("transcript_indices_active", [True] * len(transcript_indices))
                        
                        for j, t_idx in enumerate(transcript_indices):
                            if t_idx == idx and transcript_active[j]:
                                results.append({
                                    "file_id": file_id,
                                    "display_name": meta["display_name"],
                                    "date": meta["date"],
                                    "distance": float(distances[0][i]),
                                    "type": "transcript",
                                    "chunk_index": j,
                                    "context": meta.get("transcript_chunks", [])[j] if j < len(meta.get("transcript_chunks", [])) else None
                                })
        
        # Sort results by distance (smaller is better)
        results.sort(key=lambda x: x["distance"])
        
        # Populate context for non-chunked summary results
        for result in results:
            if result["type"] == "summary" and result["context"] is None:
                with open(result["file_id"], 'r') as f:
                    content = f.read()
                    summary_match = re.search(r'## Summary\s+(.*?)(?=\Z|\n## )', content, re.DOTALL)
                    if summary_match:
                        result["context"] = summary_match.group(1)
        
        return results
    
    def reindex_all(self, meeting_files: List[Dict[str, Any]]) -> None:
        """Reindex all documents."""
        logger.info("Reindexing all documents")
        
        # Reset indices
        self.summary_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.transcript_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
        self.metadata = {}
        
        # Process each file
        for file_info in meeting_files:
            try:
                file_path = file_info["path"]
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract summary and transcript with more robust regex patterns
                summary = ""
                transcript = ""
                
                # Try different summary patterns
                summary_patterns = [
                    r'## Summary\s+(.*?)(?=\Z|\n## )',
                    r'### Meeting Summary\s+(.*?)(?=\Z|\n###)',
                    r'### Consolidated Summary\s+(.*?)(?=\Z|\n###)',
                    r'### Summary\s+(.*?)(?=\Z|\n###)'
                ]
                
                for pattern in summary_patterns:
                    summary_match = re.search(pattern, content, re.DOTALL)
                    if summary_match:
                        summary = summary_match.group(1).strip()
                        break
                
                # Try different transcript patterns
                transcript_patterns = [
                    r'## Full Transcript\s+```\s+(.*?)\s+```',
                    r'### Full Transcript\s+```\s+(.*?)\s+```',
                    r'## Transcript\s+```\s+(.*?)\s+```',
                    r'### Transcript\s+```\s+(.*?)\s+```'
                ]
                
                for pattern in transcript_patterns:
                    transcript_match = re.search(pattern, content, re.DOTALL)
                    if transcript_match:
                        transcript = transcript_match.group(1).strip()
                        break
                
                # Log what we found
                logger.info(f"   Found summary: {summary[:50]}")
                logger.info(f"   Found transcript: {transcript[:50]}")
                
                # Index document
                self.index_document(file_info, summary, transcript)
                
            except Exception as e:
                logger.error(f"Error indexing {file_info['path']}: {str(e)}")
        
        # Save indices
        self.save_indices()

# Helper function to get semantic search engine instance
def get_search_engine() -> SemanticSearchEngine:
    """Get or create a semantic search engine instance."""
    if not hasattr(get_search_engine, "instance"):
        get_search_engine.instance = SemanticSearchEngine()
    return get_search_engine.instance