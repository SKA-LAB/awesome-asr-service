from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from app.worker import transcribe_audio_task
import uuid
import os

app = FastAPI()

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if file.content_type != "audio/mpeg":
        return JSONResponse(status_code=400, content={"error": "Only MP3 files are supported."})

    task_id = str(uuid.uuid4())
    temp_path = f"/tmp/{task_id}.mp3"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    # Launch transcription task
    transcribe_audio_task.delay(temp_path, task_id)

    return {"message": "Transcription started", "task_id": task_id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result_path = f"/tmp/{task_id}.txt"
    if os.path.exists(result_path):
        with open(result_path) as f:
            return {"transcription": f.read()}
    else:
        return {"status": "processing"}
