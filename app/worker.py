from celery import Celery
from app.utils import convert_and_transcribe
import os

app = Celery("worker", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")

@app.task
def transcribe_audio_task(mp3_path: str, task_id: str):
    try:
        transcription = convert_and_transcribe(mp3_path)
        result_path = f"/tmp/{task_id}.txt"
        with open(result_path, "w") as f:
            f.write(transcription)
    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)
