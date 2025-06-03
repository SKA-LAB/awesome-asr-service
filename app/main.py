from fastapi import FastAPI, File, UploadFile, Body, Depends, HTTPException
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

TEMP_DIR = os.environ.get("TEMP_DIR", "/tmp")

# docker build -t asr-service .
# docker run -p 8585:8000 -p 8586:8501 --rm asr-service

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
    
    if file.content_type != "audio/mpeg":
        logger.warning(f"Invalid file type: {file.content_type}")
        return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})
    
    task_id = str(uuid.uuid4())
    temp_path = f"{TEMP_DIR}/{task_id}.mp3"
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
                         model="qwen2.5:7b",
                         api_key="null",
                         base_url="http://localhost:11434")
            
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
You are an expert meeting assistant. Based on these summaries of different parts of a meeting transcript, 
create a cohesive final summary with two clear sections:
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
You are an expert meeting assistant. Summarize the following transcript and extract action items or to-dos.
Respond in well-formatted Markdown with two sections. Add sub-sections as needed for the summary and action items:
1. ## Meeting Summary
2. ## Action Items (as a bullet list)
    
Transcript:
{transcript}
"""
            logger.info("Initializing Ollama LLM")
            llm = ChatOllama(temperature=0.2,
                         model="qwen2.5:7b",
                         api_key="null",
                         base_url="http://localhost:11434")
            
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
async def transcribe_and_summarize(request: Request, file: UploadFile = File(...)):
    logger.info(f"Transcribe and summarize request received for file: {file.filename}")
    
    if file.content_type != "audio/mpeg":
        logger.warning(f"Invalid file type: {file.content_type}")
        return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})

    task_id = str(uuid.uuid4())
    temp_path = f"{TEMP_DIR}/{task_id}.mp3"
    logger.info(f"Created task ID: {task_id}")
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
            logger.info(f"Saved file to {temp_path}, size: {len(content)} bytes")
            
        # Use the synchronous version for immediate processing
        logger.info("Starting synchronous transcription")
        from app.utils import convert_and_transcribe
        transcription = convert_and_transcribe(temp_path)
        logger.info(f"Transcription completed, length: {len(transcription)} chars")
        
        # Now summarize the transcription
        logger.info("Starting summarization")
        prompt = f"""
You are an expert meeting assistant. Summarize the following transcript and extract action items or to-dos.
Respond in well-formatted Markdown with two sections. Add sub-sections as needed for the summary and action items:
1. ## Meeting Summary
2. ## Action Items (as a bullet list)
    
Transcript:
{transcription}
"""
        logger.info("Initializing Ollama LLM")
        llm = ChatOllama(temperature=0.2,
                         model="qwen2.5:7b",
                         api_key="null",
                         base_url="http://localhost:11434")
        
        logger.info("Sending request to Ollama")
        response = llm.invoke(prompt)
        markdown_output = response.content.strip()
        logger.info(f"Received summary from Ollama, length: {len(markdown_output)} chars")
        
        return {
            "transcription": transcription,
            "summary_markdown": markdown_output
        }
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}", exc_info=True)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")

@app.get("/health")
@limiter.limit("10/minute")
async def health_check(request: Request):
    logger.info("Health check requested")
    try:
        # Check if Ollama service is available
        logger.info("Testing Ollama service")
        llm = ChatOllama(model="qwen2.5:7b", api_key="null", base_url="http://localhost:11434")
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)