import os
import logging
from typing import List, Dict, Any
import re
from datetime import datetime
from semantic_search import get_search_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("indexing-utils")

def index_meeting_files(meeting_files: List[Dict[str, Any]], force_reindex: bool = False) -> None:
    """
    Index meeting files for semantic search.
    
    Args:
        meeting_files: List of meeting file information
        force_reindex: If True, reindex all files even if they haven't changed
    """
    search_engine = get_search_engine()
    
    if force_reindex:
        logger.info("Force reindexing all meeting files")
        search_engine.reindex_all(meeting_files)
        return
    
    # Check if any files need indexing
    for file_info in meeting_files:
        file_path = file_info["path"]
        
        # Check if file is already indexed and unchanged
        if file_path in search_engine.metadata:
            last_modified = os.path.getmtime(file_path)
            if last_modified <= search_engine.metadata[file_path].get("last_modified", 0):
                logger.debug(f"File already indexed and unchanged: {file_info['display_name']}")
                continue
        
        # File needs indexing
        logger.info(f"Indexing file: {file_info['display_name']}")
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Extract summary and transcript
            summary_match = re.search(r'## Summary\s+(.*?)(?=\Z|\n## )', content, re.DOTALL)
            summary = summary_match.group(1) if summary_match else ""
            
            transcript_match = re.search(r'## Full Transcript\s+```\s+(.*?)\s+```', content, re.DOTALL)
            transcript = transcript_match.group(1) if transcript_match else ""
            
            # Index document
            search_engine.index_document(file_info, summary, transcript)
            
        except Exception as e:
            logger.error(f"Error indexing {file_path}: {str(e)}")
    
    # Save indices
    search_engine.save_indices()

def perform_semantic_search(
    query: str, 
    files: List[Dict[str, Any]], 
    search_in_summaries: bool = True,
    search_in_transcripts: bool = True,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on meeting files.
    
    Args:
        query: Search query
        files: List of meeting file information to search in
        search_in_summaries: Whether to search in summaries
        search_in_transcripts: Whether to search in transcripts
        top_k: Number of top results to return
        
    Returns:
        List of search results
    """
    search_engine = get_search_engine()
    
    # Get semantic search results
    semantic_results = search_engine.search(
        query, 
        search_in_summaries=search_in_summaries,
        search_in_transcripts=search_in_transcripts,
        top_k=top_k
    )
    
    # Filter results to only include files in the provided list
    if files:
        file_paths = {file["path"] for file in files}
        filtered_results = [result for result in semantic_results if result["file_id"] in file_paths]
    else:
        filtered_results = semantic_results
    
    # Format results for display
    formatted_results = []
    for result in filtered_results:
        # Find the corresponding file_info
        file_info = next((file for file in files if file["path"] == result["file_id"]), None)
        if not file_info:
            continue
            
        # Format the result
        formatted_result = {
            "file_info": file_info,
            "summary_matches": [],
            "transcript_matches": [],
            "match_count": 1,  # Each semantic result is one match
            "semantic_score": 1.0 - min(1.0, result["distance"] / 2.0)  # Convert distance to a score between 0-1
        }
        
        # Add the match to the appropriate list
        if result["type"] == "summary":
            formatted_result["summary_matches"].append({
                "context": result["context"],
                "type": "summary",
                "semantic_score": formatted_result["semantic_score"]
            })
        else:  # transcript
            formatted_result["transcript_matches"].append({
                "context": result["context"],
                "type": "transcript",
                "semantic_score": formatted_result["semantic_score"]
            })
        
        formatted_results.append(formatted_result)
    
    # Combine results for the same file
    combined_results = {}
    for result in formatted_results:
        file_path = result["file_info"]["path"]
        
        if file_path not in combined_results:
            combined_results[file_path] = result
        else:
            # Update match count
            combined_results[file_path]["match_count"] += result["match_count"]
            
            # Combine matches
            combined_results[file_path]["summary_matches"].extend(result["summary_matches"])
            combined_results[file_path]["transcript_matches"].extend(result["transcript_matches"])
            
            # Update semantic score to the highest score
            combined_results[file_path]["semantic_score"] = max(
                combined_results[file_path]["semantic_score"],
                result["semantic_score"]
            )
    
    # Convert back to list and sort by semantic score
    result_list = list(combined_results.values())
    result_list.sort(key=lambda x: x["semantic_score"], reverse=True)
    
    return result_list