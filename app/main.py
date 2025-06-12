from fastapi import FastAPI, File, UploadFile, Body
from starlette.requests import Request
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse
from langchain_ollama import ChatOllama
from app.worker import transcribe_audio_task
import uuid
import os
import logging
import time
from fastapi.middleware.cors import CORSMiddleware
from app.notion_integration import NotionIntegration
from semantic_search import get_search_engine
from rag_search import get_rag_search_engine
from datetime import datetime
from typing import Optional, List

TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/meeting-notes")

# docker build -t asr-service .
# docker run -p 8585:8000 -p 8586:8501 -p 8587:8502 -v /Users/kbhattacha/Documents/meeting-notes:/meeting-notes -v /Users/kbhattacha/Music/Piezo:/meeting-recordings --name asr-service-local --rm asr-service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logger = logging.getLogger("asr-service")

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="ASR Service API",
    description="API for transcribing audio files and summarizing meeting transcripts",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add after app initialization
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["localhost", "127.0.0.1"]  # Add your production domains
)

# Add middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    start_time = time.time()
    
    # Generate request ID for tracking
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id} started: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Request {request_id} completed: {response.status_code} in {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise

# Add rate limiting to endpoints
@app.post("/transcribe/")
@limiter.limit("10/minute")
async def transcribe(request: Request, file: UploadFile = File(...)):
    logger.info(f"Transcription request received for file: {file.filename}")
    
    if file.content_type not in ["audio/mpeg", "audio/mp4", "audio/x-m4a"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        return JSONResponse(status_code=400, content={"error": "Only MP3 and M4A files are supported."})
    
    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.mp3', '.m4a']:
        file_extension = '.mp3' if file.content_type == "audio/mpeg" else '.m4a'
    
    temp_path = f"{TEMP_DIR}/{task_id}{file_extension}"
    logger.info(f"Created task ID: {task_id}")
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info(f"Saved file to {temp_path}, size: {len(content)} bytes")
            
        # Launch transcription task
        transcribe_audio_task.delay(temp_path, task_id)
        logger.info(f"Transcription task {task_id} queued successfully")
        
        return {"message": "Transcription started", "task_id": task_id}
    except Exception as e:
        logger.error(f"Failed to process file: {str(e)}", exc_info=True)
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")
        return JSONResponse(status_code=500, content={"error": f"Failed to process file: {str(e)}"})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    logger.info(f"Result requested for task: {task_id}")
    result_path = f"{TEMP_DIR}/{task_id}.txt"
    
    if os.path.exists(result_path):
        try:
            with open(result_path) as f:
                content = f.read()
                logger.info(f"Result found for task {task_id}, length: {len(content)} chars")
                os.remove(result_path)
                logger.info(f"Cleaned up result file: {result_path}")
                return {"transcription": content}
        except Exception as e:
            logger.error(f"Failed to read result for task {task_id}: {str(e)}", exc_info=True)
            return JSONResponse(status_code=500, content={"error": f"Failed to read result: {str(e)}"})
    else:
        logger.info(f"Result for task {task_id} not ready yet")
        return JSONResponse(status_code=202, content={"status": "processing"})
    

@app.post("/summarize/")
@limiter.limit("10/minute")
async def summarize_meeting(request: Request, transcript: str = Body(..., embed=True)):
    logger.info("Summarization request received")
    
    if not transcript or transcript.strip() == "":
        logger.warning("Empty transcript provided")
        return JSONResponse(status_code=400, content={"error": "Empty transcript provided"})
    
    try:
        transcript_length = len(transcript)
        logger.info(f"Processing transcript of length: {transcript_length} chars")
        
        # Check if transcript is very long
        if transcript_length > 10000:  # Adjust this threshold based on your model's capabilities
            logger.info(f"Long transcript detected ({transcript_length} chars). Using chunking approach.")
            
            # Split transcript into chunks
            chunk_size = 8000  # Adjust based on model context window
            overlap = 500  # Overlap between chunks to maintain context
            chunks = []
            
            for i in range(0, transcript_length, chunk_size - overlap):
                chunk = transcript[i:i + chunk_size]
                chunks.append(chunk)
                
            logger.info(f"Split transcript into {len(chunks)} chunks")
            
            # Process each chunk
            chunk_summaries = []
            llm = ChatOllama(temperature=0.2,
                         model="qwen2.5:3b",
                         api_key="null",
                         base_url=OLLAMA_BASE_URL)
            
            for i, chunk in enumerate(chunks):
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
                chunk_prompt = f"""
Summarize this part of a meeting transcript. Include all important discussion points, names, places, and activities mentioned. 

{chunk}
"""
                response = llm.invoke(chunk_prompt)
                chunk_summaries.append(response.content.strip())
                
            # Combine chunk summaries and create final summary
            combined_summary = "\n\n".join(chunk_summaries)
            logger.info("Creating final summary from chunk summaries")
            
            final_prompt = f"""
You are an expert meeting assistant. Your task is to review the combined summaries of meeting transcripts and create a consolidated summary that captures the essential information,
focusing on key takeaways and action items assigned to specific individuals or departments during the meeting.
Use clear and professional language, and organize the summary in a logical manner using appropriate formatting such as
headings, subheadings, and bullet points. Ensure that the summary is easy to understand and provides a comprehensive but succinct
overview of the meeting's content, with a particular focus on clearly indicating who is responsible for each action item.

1. ## Meeting Summary
2. ## Action Items (as a bullet list)

Make sure to consolidate duplicate information and present a unified view.

Summaries from different parts of the transcript:
{combined_summary}
"""
            response = llm.invoke(final_prompt)
            markdown_output = response.content.strip()
            
        else:
            # Original approach for shorter transcripts
            prompt = f"""
Your task is to review the provided meeting notes and create a concise summary that captures the essential information,
focusing on key takeaways and action items assigned to specific individuals or departments during the meeting. Use clear
and professional language, and organize the summary in a logical manner using appropriate formatting such as headings,
subheadings, and bullet points. Ensure that the summary is easy to understand and provides a comprehensive but succinct
overview of the meeting's content, with a particular focus on clearly indicating who is responsible for each action item.
Respond in well-formatted Markdown with two sections. Add sub-sections as needed for the summary and action items:
1. ## Meeting Summary
2. ## Action Items (as a bullet list)
    
Transcript:
{transcript}
"""
            logger.info("Initializing Ollama LLM")
            llm = ChatOllama(temperature=0.2,
                         model="qwen2.5:3b",
                         api_key="null",
                         base_url=OLLAMA_BASE_URL)
            
            logger.info("Sending request to Ollama")
            response = llm.invoke(prompt)
            markdown_output = response.content.strip()
            
        logger.info(f"Received summary, length: {len(markdown_output)} chars")
        return {"summary_markdown": markdown_output}
    except Exception as e:
        logger.error(f"Summarization failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Summarization failed: {str(e)}"})

@app.post("/transcribe-and-summarize/")
@limiter.limit("10/minute")
async def transcribe_and_summarize(
    request: Request, 
    file: UploadFile = File(...), 
    save_to_file: bool = False,
    save_to_notion: bool = False,
    meeting_date: Optional[str] = None,
    participants: Optional[List[str]] = None
):
    logger.info(f"Transcribe and summarize request received for file: {file.filename}")
    
    if file.content_type not in ["audio/mpeg", "audio/mp4", "audio/x-m4a"]:
        logger.warning(f"Invalid file type: {file.content_type}")
        return JSONResponse(status_code=400, content={"error": "Only MP3 and M4A files are supported."})

    task_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in ['.mp3', '.m4a']:
        file_extension = '.mp3' if file.content_type == "audio/mpeg" else '.m4a'
    
    temp_path = f"{TEMP_DIR}/{task_id}{file_extension}"
    logger.info(f"Created task ID: {task_id}")
    
    try:
        # Save uploaded file to temp location
        await save_uploaded_file(file, temp_path)
        
        # Process the audio file
        title, transcription, markdown_output = await process_audio_file(temp_path, file.filename)
        
        # Save results if requested
        file_path = save_to_file_system(save_to_file, OUTPUT_DIR, file.filename, transcription, markdown_output)
        
        # Save to Notion if requested
        notion_page_url = await save_to_notion_db(save_to_notion, title, markdown_output, transcription, meeting_date, participants)
        
        return {
            "title": title,
            "transcription": transcription,
            "summary_markdown": markdown_output,
            "saved_to_file": file_path if save_to_file else None,
            "saved_to_notion": notion_page_url
        }
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": f"Processing failed: {str(e)}"})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")

async def save_uploaded_file(file: UploadFile, temp_path: str) -> None:
    """Save the uploaded file to a temporary location."""
    with open(temp_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
        logger.info(f"Saved file to {temp_path}, size: {len(content)} bytes")

async def process_audio_file(temp_path: str, filename: str) -> tuple:
    """Process the audio file to get transcription and summary."""
    # Transcribe the audio
    logger.info("Starting synchronous transcription")
    from app.utils import convert_and_transcribe
    transcription = convert_and_transcribe(temp_path)
    logger.info(f"Transcription completed, length: {len(transcription)} chars")
    
    # Summarize the transcription
    logger.info("Starting summarization")
    markdown_output = await generate_summary(transcription)
    
    # Create a title from the filename
    title = f"{filename.split('.')[0]}"
    
    return title, transcription, markdown_output

async def generate_summary(transcription: str) -> str:
    """
    Generate a summary from the transcription using Ollama.
    For long transcripts, uses a chunking approach to handle context window limitations.
    """
    transcript_length = len(transcription)
    logger.info(f"Processing transcript of length: {transcript_length} chars")
    
    # Initialize LLM
    logger.info("Initializing Ollama LLM")
    llm = ChatOllama(temperature=0.2,
                     model="qwen2.5:3b",
                     api_key="null",
                     base_url=OLLAMA_BASE_URL)
    
    # Check if transcript is very long
    if transcript_length > 10000:  # Adjust this threshold based on model's capabilities
        logger.info(f"Long transcript detected ({transcript_length} chars). Using chunking approach.")
        return await process_long_transcript(transcription, llm)
    else:
        logger.info("Using standard approach for shorter transcript")
        return await process_standard_transcript(transcription, llm)

async def process_long_transcript(transcription: str, llm: ChatOllama) -> str:
    """Process a long transcript using chunking approach."""
    # Split transcript into chunks
    chunk_size = 8000  # Adjust based on model context window
    overlap = 500  # Overlap between chunks to maintain context
    transcript_length = len(transcription)
    chunks = []
    
    for i in range(0, transcript_length, chunk_size - overlap):
        chunk = transcription[i:i + chunk_size]
        chunks.append(chunk)
        
    logger.info(f"Split transcript into {len(chunks)} chunks")
    
    # Process each chunk
    chunk_summaries = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"Processing chunk {i+1}/{len(chunks)}")
        chunk_prompt = f"""
Summarize this part of a meeting transcript. Focus on key points and any action items mentioned:

{chunk}
"""
        response = llm.invoke(chunk_prompt)
        chunk_summaries.append(response.content.strip())
        
    # Combine chunk summaries and create final summary
    combined_summary = "\n\n".join(chunk_summaries)
    logger.info("Creating final summary from chunk summaries")
    
    final_prompt = f"""
You are an expert meeting assistant. Your task is to review the combined summaries of meeting transcripts and create a consolidated summary that captures the essential information,
focusing on key takeaways and action items assigned to specific individuals or departments during the meeting.
Use clear and professional language, and organize the summary in a logical manner using appropriate formatting such as
headings, subheadings, and bullet points. Ensure that the summary is easy to understand and provides a comprehensive but succinct
overview of the meeting's content, with a particular focus on clearly indicating who is responsible for each action item.

1. ## Meeting Summary
2. ## Action Items (as a bullet list)

Make sure to consolidate duplicate information and present a unified view.

Summaries from different parts of the transcript:
{combined_summary}
"""
    response = llm.invoke(final_prompt)
    markdown_output = response.content.strip()
    logger.info(f"Received consolidated summary, length: {len(markdown_output)} chars")
    
    return markdown_output

async def process_standard_transcript(transcription: str, llm: ChatOllama) -> str:
    """Process a standard-length transcript."""
    prompt = f"""
Your task is to review the provided meeting notes and create a concise summary that captures the essential information,
focusing on key takeaways and action items assigned to specific individuals or departments during the meeting. Use clear
and professional language, and organize the summary in a logical manner using appropriate formatting such as headings,
subheadings, and bullet points. Ensure that the summary is easy to understand and provides a comprehensive but succinct
overview of the meeting's content, with a particular focus on clearly indicating who is responsible for each action item.
Respond in well-formatted Markdown with two sections. Add sub-sections as needed for the summary and action items:
1. ## Meeting Summary
2. ## Action Items (as a bullet list)
    
Transcript:
{transcription}
"""
    logger.info("Sending request to Ollama")
    response = llm.invoke(prompt)
    markdown_output = response.content.strip()
    logger.info(f"Received summary from Ollama, length: {len(markdown_output)} chars")
    
    return markdown_output

def save_to_file_system(save_to_file: bool, output_dir: str, filename: str, transcription: str, markdown_output: str) -> Optional[str]:
    """Save the transcription and summary to a file if requested."""
    if not save_to_file:
        return None
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a filename based on the original file and timestamp
    base_filename = filename.split('.')[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = f"{output_dir}/{base_filename}_{timestamp}.md"
    
    # Write the content to the file
    with open(file_path, "w") as f:
        f.write(f"# {base_filename} - Transcript and Summary\n\n")
        f.write("## Full Transcript\n\n")
        f.write("```\n")
        f.write(transcription)
        f.write("\n```\n\n")
        f.write("## Summary\n\n")
        f.write(markdown_output)
    
    logger.info(f"Saved transcript and summary to {file_path}")
    return file_path

async def save_to_notion_db(save_to_notion: bool, title: str, summary: str, transcript: str, 
                           meeting_date: Optional[str], participants: Optional[List[str]]) -> Optional[str]:
    """Save the transcription and summary to Notion if requested."""
    if not save_to_notion:
        return None
        
    try:
        logger.info("Saving to Notion")
        notion = NotionIntegration()
        
        # Parse meeting date if provided, otherwise use current date
        meeting_datetime = None
        if meeting_date:
            try:
                meeting_datetime = datetime.strptime(meeting_date, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Invalid meeting date format: {meeting_date}. Using current date.")
                meeting_datetime = datetime.now()
        else:
            meeting_datetime = datetime.now()
        
        # Create the page in Notion
        notion_response = notion.create_meeting_page(
            title=title,
            summary=summary,
            transcript=transcript,
            meeting_date=meeting_datetime,
            participants=participants
        )
        
        notion_page_url = notion_response.get("url")
        logger.info(f"Successfully saved to Notion: {notion_page_url}")
        return notion_page_url
    except Exception as e:
        logger.error(f"Failed to save to Notion: {str(e)}")
        # Don't fail the whole request if Notion integration fails
        return None

@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    logger.info("Health check requested")
    try:
        # Check if Ollama service is available
        logger.info("Testing Ollama service")
        llm = ChatOllama(model="qwen2.5:3b", api_key="null", base_url=OLLAMA_BASE_URL)
        llm.invoke("Hello")
        logger.info("Ollama service is available")
        
        # Check if Redis is available for Celery
        logger.info("Testing Redis/Celery connection")
        from app.worker import app as celery_app
        celery_app.control.ping()
        logger.info("Redis is available")
        
        return {"status": "healthy", "services": {"ollama": "available", "redis": "available"}}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})

@app.get("/semantic-search/indexed-files")
@limiter.limit("20/minute")
async def get_indexed_files(request: Request):
    """
    Returns all filenames currently indexed in the FAISS database.
    """
    logger.info("Request for indexed files received")
    try:
        search_engine = get_search_engine()
        
        # Extract filenames from metadata
        indexed_files = []
        for file_id, meta in search_engine.metadata.items():
            indexed_files.append({
                "file_id": file_id,
                "display_name": meta.get("display_name", os.path.basename(file_id)),
                "last_modified": datetime.fromtimestamp(meta.get("last_modified", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                "summary_indexed": meta.get("summary_idx_active", False),
                "transcript_indexed": any(meta.get("transcript_indices_active", [])),
            })
        
        return {
            "success": True,
            "indexed_files": indexed_files,
            "summary_count": search_engine.summary_index.ntotal if search_engine.summary_index else 0,
            "transcript_chunk_count": search_engine.transcript_index.ntotal if search_engine.transcript_index else 0
        }
    except Exception as e:
        logger.error(f"Failed to retrieve indexed files: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to retrieve indexed files: {str(e)}"})

@app.post("/semantic-search/rebuild-index")
@limiter.limit("5/minute")
async def rebuild_search_index(request: Request):
    """
    Clears the FAISS index and re-indexes all meeting files.
    """
    logger.info("Request to rebuild search index received")
    try:
        # Load all meeting files
        from meeting_explorer import load_meeting_files
        meeting_files = load_meeting_files()
        
        if not meeting_files:
            return JSONResponse(status_code=404, content={"success": False, "error": "No meeting files found to index"})
        
        # Get search engine and rebuild index
        search_engine = get_search_engine()
        search_engine.reindex_all(meeting_files)
        
        return {
            "success": True,
            "message": f"Search index rebuilt successfully with {len(meeting_files)} files",
            "indexed_files_count": len(meeting_files),
            "summary_count": search_engine.summary_index.ntotal,
            "transcript_chunk_count": search_engine.transcript_index.ntotal
        }
    except Exception as e:
        logger.error(f"Failed to rebuild search index: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Failed to rebuild search index: {str(e)}"})

@app.get("/semantic-search/search")
@limiter.limit("20/minute")
async def semantic_search(
    request: Request, 
    query: str,
    search_in_summaries: bool = True,
    search_in_transcripts: bool = True,
    top_k: int = 5
):
    """
    Performs a semantic search using the FAISS index and returns results based on the query.
    """
    logger.info(f"Semantic search request received for query: {query}")
    
    if not query or query.strip() == "":
        return JSONResponse(status_code=400, content={"success": False, "error": "Query cannot be empty"})
    
    try:
        # Import the function from indexing_utils
        from app.indexing_utils import perform_semantic_search
        
        # Perform semantic search
        results = perform_semantic_search(
            query=query,
            search_in_summaries=search_in_summaries,
            search_in_transcripts=search_in_transcripts,
            top_k=top_k
        )
        
        return {
            "success": True,
            "query": query,
            "results": results,
            "result_count": len(results)
        }
    except Exception as e:
        logger.error(f"Semantic search failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"Semantic search failed: {str(e)}"})

@app.post("/rag-search")
@limiter.limit("10/minute")
async def rag_search(
    request: Request,
    query: str = Body(...),
    conversation_history: Optional[List[dict]] = Body(None),
    limit_files: Optional[List[str]] = Body(None)
):
    """
    Performs a RAG search with optional conversation history using the RAGSearchEngine.
    
    conversation_history format: [{"role": "user", "content": "message"}, {"role": "assistant", "content": "response"}]
    """
    logger.info(f"RAG search request received for query: {query}")
    
    if not query or query.strip() == "":
        return JSONResponse(status_code=400, content={"success": False, "error": "Query cannot be empty"})
    
    try:
        # Get RAG search engine with optional file limits
        rag_engine = get_rag_search_engine(limit_files)
        
        # Format conversation history if provided
        formatted_history = None
        if conversation_history:
            formatted_history = []
            for msg in conversation_history:
                if "role" in msg and "content" in msg:
                    formatted_history.append((msg["role"], msg["content"]))
        
        # Perform RAG search with history
        response = rag_engine.search_with_history(query, formatted_history)
        
        return {
            "success": response["success"],
            "answer": response["answer"],
            "context_used": response["context_used"]
        }
    except Exception as e:
        logger.error(f"RAG search failed: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"success": False, "error": f"RAG search failed: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)