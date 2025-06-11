import os
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import logging
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from meeting_reranker import MeetingNotesReRanker
from indexing_utils import perform_semantic_search
from action_items_extractor import ActionItemsExtractor
from timeit import default_timer as timer
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-search")

# Constants
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
MAX_CONTEXT_CHUNKS = 5

class RAGSearchEngine:
    def __init__(self, limit_files: List[str] = None):
        self.llm = ChatOllama(base_url=OLLAMA_BASE_URL,
                              model=LLM_MODEL,
                              temperature=0.5,
                              num_ctx=100000)
        self.reranker = MeetingNotesReRanker()
        self.action_items_extractor = ActionItemsExtractor()
        self.search_files = limit_files
    
    def _create_agent(self):
        """Create a ReAct agent for RAG search."""
        # Define the tools the agent can use
        tools = [
            self._search_summaries,
            self._search_transcripts,
            self._extract_action_items,
        ]
        
        # Create the ReAct agent
        system_message = f"""You are an intelligent meeting assistant that helps users find information in meeting notes and answer general questions.
You maintain conversation context and can refer to previous messages in the conversation.
Use the search tools to find relevant information in meeting summaries and transcripts.
Use the action items extractor to identify action items defined during meetings.
Go deep in your search to find the most relevant information.
You may have to do follow-up searches with new queries based on the results of previous searches.
Always provide specific and relevant answers based on the retrieved information.
If you can't find relevant information, admit that you don't know.
When responding, include your answer, rationale, and source materials used.
When referring to previous parts of the conversation, be specific about what was discussed.
Today's date is {datetime.now().strftime('%Y-%m-%d')}, use this information as needed to contextualize dates and times in meetings that may have happened in the past."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{messages}"),
        ])
        
        agent = create_react_agent(self.llm, tools,
                                   state_modifier=prompt)
        return agent
    
    def _extract_action_items(self, date_range: str = "") -> str:
        """
        Extract action items from meeting notes within a date range.
        
        Args:
            date_range: Optional string specifying the date range in format 'YYYY-MM-DD to YYYY-MM-DD'
                        If not provided, all meetings will be included
        
        Returns:
            Formatted string of action items
        """
        logger.info(f"Extracting action items with date range: {date_range}")

        # Parse date range if provided
        start_date = None
        end_date = None
        
        if date_range:
            try:
                # Try to parse different date range formats
                if " to " in date_range:
                    start_str, end_str = date_range.split(" to ")
                    start_date = datetime.strptime(start_str.strip(), "%Y-%m-%d")
                    end_date = datetime.strptime(end_str.strip(), "%Y-%m-%d")
                elif "-" in date_range and len(date_range.split("-")) == 3:
                    # Single date
                    single_date = datetime.strptime(date_range.strip(), "%Y-%m-%d")
                    start_date = single_date
                    end_date = single_date
            except ValueError:
                return "Invalid date range format. Please use 'YYYY-MM-DD to YYYY-MM-DD' format."
            
        # Extract action items
        action_items_text = self.action_items_extractor.extract_from_files(self.search_files, start_date, end_date)
        
        return action_items_text
    
    def _search_summaries(self, query: str) -> str:
        """Search in meeting summaries with a query."""
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
        """Search in meeting transcripts with a query."""
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
            # Perform semantic search
            initial_k = 30  # initially retrieve a large number of results that will be reranked
            tic = timer()
            result_list = perform_semantic_search(
                query,
                self.search_files,
                search_in_summaries=search_in_summaries,
                search_in_transcripts=search_in_transcripts,
                top_k=initial_k
            )
            toc = timer()
            logger.info(f"Semantic search took {toc - tic:.2f} seconds")

            # Apply meeting-specific re-ranking
            tic = timer()
            reranked_results = self.reranker.rerank(
                query=query,
                results=result_list,
                top_k=MAX_CONTEXT_CHUNKS
            )
            toc = timer()
            logger.info(f"Meeting-specific re-ranking took {toc - tic:.2f} seconds")
            
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error performing search: {str(e)}")
            return []
        
    def search_with_history(self, query: str, history: List[tuple]=None) -> Dict[str, Any]:
        """
        Perform a RAG-based search using the ReAct agent with conversation history.
        
        Args:
            query: The user's current search query
            history: List of (role, content) tuples representing the conversation history
            
        Returns:
            Dictionary containing the agent's response and search results
        """
        logger.info(f"RAG search with history for query: {query}")
        
        try:
            # Format the conversation history for the agent
            messages = []
            if history:
                for role, content in history:
                    messages.append((role, content))
            
            # Add the current query
            messages.append(('user', query))
            
            # Run the agent with conversation history
            inputs = {"messages": messages}
            agent = self._create_agent()
            response = agent.invoke(inputs, {"recursion_limit": 14})
            
            # Grab all context used to generate the final answer
            contexts = []
            for message in response["messages"]:
                if type(message) == ToolMessage:
                    contexts.append(message.content)
            contexts = "\n\n".join(contexts)
            
            return {
                "answer": response["messages"][-1].content,
                "context_used": contexts,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error in RAG search with history: {str(e)}")
            return {
                "answer": f"Sorry, I encountered an error while searching: {str(e)}",
                "context_used": "",
                "success": False
            }

# Singleton instance
_rag_search_instance = None

def get_rag_search_engine(search_files: List[str]=None) -> RAGSearchEngine:
    """Get or create the RAG search engine instance."""
    global _rag_search_instance
    if _rag_search_instance is None:
        _rag_search_instance = RAGSearchEngine(search_files)
    _rag_search_instance.search_files = search_files  # Update search files if provided
    return _rag_search_instance