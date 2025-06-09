import os
from typing import List, Dict, Any, BaseModel, Field
import logging
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from semantic_search import get_search_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-search")

# Constants
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
MAX_CONTEXT_CHUNKS = 5

class RAGResponseFormat(BaseModel):
    answer: str = Field(description="The final answer provided with structure and detail.")
    rationale: str = Field(description="The rationale used to arrive at the answer based on the retrieved information.")
    snippets: List[str] = Field(description="List of short relevant snippets from meeting used to arrive at the answer.")

class RAGSearchEngine:
    def __init__(self):
        self.llm = ChatOllama(base_url=OLLAMA_BASE_URL,
                              model=LLM_MODEL,
                              temperature=0.5,
                              num_ctx=100000)
        self.search_engine = get_search_engine()
        self.agent = self._create_agent()
    
    def _create_agent(self):
        """Create a ReAct agent for RAG search."""
        # Define the tools the agent can use
        tools = [
            {
                "name": "search_summaries",
                "description": "Search in meeting summaries for relevant information",
                "fn": self._search_summaries,
                "args_schema": {"query": str}
            },
            {
                "name": "search_transcripts",
                "description": "Search in meeting transcripts for detailed information",
                "fn": self._search_transcripts,
                "args_schema": {"query": str}
            }
        ]
        
        # Create the ReAct agent
        system_message = """You are an intelligent meeting assistant that helps users find information in meeting notes.
        Use the search tools to find relevant information in meeting summaries and transcripts.
        Always provide specific and concise answers based on the retrieved information.
        If you can't find relevant information, admit that you don't know.
        When responding, including your answer, rationale, and source materials used."""
        
        agent = create_react_agent(self.llm, tools,
                                   state_modifier=system_message,
                                   response_format=RAGResponseFormat)
        return agent
    
    def _search_summaries(self, query: str) -> str:
        """Search in meeting summaries."""
        logger.info(f"Searching summaries for: {query}")
        results = self._perform_search(query, search_in_summaries=True, search_in_transcripts=False)
        
        if not results:
            return "No relevant information found in meeting summaries."
        
        # Format results for the agent
        formatted_results = []
        for result in results[:MAX_CONTEXT_CHUNKS]:
            meeting_name = result["file_info"]["display_name"]
            for match in result["summary_matches"]:
                formatted_results.append(f"From meeting '{meeting_name}':\n{match['context']}")
        
        return "\n\n".join(formatted_results)
    
    def _search_transcripts(self, query: str) -> str:
        """Search in meeting transcripts."""
        logger.info(f"Searching transcripts for: {query}")
        results = self._perform_search(query, search_in_summaries=False, search_in_transcripts=True)
        
        if not results:
            return "No relevant information found in meeting transcripts."
        
        # Format results for the agent
        formatted_results = []
        for result in results[:MAX_CONTEXT_CHUNKS]:
            meeting_name = result["file_info"]["display_name"]
            for match in result["transcript_matches"]:
                formatted_results.append(f"From meeting '{meeting_name}':\n{match['context']}")
        
        return "\n\n".join(formatted_results)
    
    def _perform_search(self, query: str, search_in_summaries: bool = True, search_in_transcripts: bool = True) -> List[Dict[str, Any]]:
        """Perform semantic search using the FAISS index."""
        try:
            # Get all indexed files
            indexed_files = []
            for file_id, meta in self.search_engine.metadata.items():
                indexed_files.append({
                    "path": file_id,
                    "display_name": meta.get("display_name", os.path.basename(file_id))
                })
            
            # Perform semantic search
            results = self.search_engine.search(
                query,
                search_in_summaries=search_in_summaries,
                search_in_transcripts=search_in_transcripts,
                top_k=MAX_CONTEXT_CHUNKS
            )
            
            # Format results similar to indexing_utils.perform_semantic_search
            formatted_results = []
            for result in results:
                # Find the corresponding file_info
                file_info = next((file for file in indexed_files if file["path"] == result["file_id"]), None)
                if not file_info:
                    continue
                    
                # Format the result
                formatted_result = {
                    "file_info": file_info,
                    "summary_matches": [],
                    "transcript_matches": [],
                    "match_count": 1,
                    "semantic_score": 1.0 - min(1.0, result["distance"] / 2.0)
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
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return []
    
    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a RAG-based search using the ReAct agent.
        
        Args:
            query: The user's search query
            
        Returns:
            Dictionary containing the agent's response and search results
        """
        logger.info(f"RAG search for query: {query}")
        
        try:
            # Run the agent
            messages = [('user', query)]
            inputs = {"messages": messages}
            messages = self.agent.invoke(inputs,
                                         {"recursion_limit": 3})
            
            # Extract the final answer
            final_answer = messages["output"]
            if "structured_response" in messages:
                if messages["structured_response"]:
                    structured_response_exists = True
            
            if structured_response_exists:
                final_answer = messages["structured_response"].answer
                rationale = messages["structured_response"].rationale
                snippets = messages["structured_response"].snippets
            else:
                final_answer = messages["messages"][-1].content
                rationale = "Not able to generate a rationale for this response."
                snippets = "Generating snippets from source documents failed for this response."
            
            return {
                "answer": final_answer,
                "thought_process": rationale,
                "snippets": snippets,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while searching: {str(e)}",
                "thought_process": "",
                "snippets": "",
                "success": False
            }

# Singleton instance
_rag_search_instance = None

def get_rag_search_engine() -> RAGSearchEngine:
    """Get or create the RAG search engine instance."""
    global _rag_search_instance
    if _rag_search_instance is None:
        _rag_search_instance = RAGSearchEngine()
    return _rag_search_instance