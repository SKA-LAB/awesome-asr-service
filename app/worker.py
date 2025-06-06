from celery import Celery
import os
import logging

app = Celery("worker", broker="redis://localhost:6379/0", backend="redis://localhost:6379/0")
logger = logging.getLogger("asr-service")

@app.task
def transcribe_audio_task(audio_path: str, task_id: str):
    try:
        logger.info(f"Starting transcription for task {task_id}")
        from app.utils import convert_and_transcribe
        
        # Perform transcription
        transcription = convert_and_transcribe(audio_path)
        
        # Save result to file
        result_path = f"{os.path.splitext(audio_path)[0]}.txt"
        with open(result_path, "w") as f:
            f.write(transcription)
        
        logger.info(f"Transcription completed for task {task_id}, saved to {result_path}")
        return {"status": "completed", "task_id": task_id}
    except Exception as e:
        logger.error(f"Transcription failed for task {task_id}: {str(e)}", exc_info=True)
        return {"status": "failed", "task_id": task_id, "error": str(e)}
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info(f"Cleaned up temporary file: {audio_path}")
