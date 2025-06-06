import os
import time
import logging
from pydub import AudioSegment
import threading
from faster_whisper import WhisperModel

# Get logger
logger = logging.getLogger("asr-service")

# Singleton pattern for ASR model
class ASRModelSingleton:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    logger.info("Loading faster_whisper model...")
                    # You can choose model size: "tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"
                    model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")
                    
                    cls._instance = WhisperModel(
                        model_size
                    )
        return cls._instance

def convert_and_transcribe(audio_path: str) -> str:
    start_time = time.time()
    logger.info(f"Starting transcription for file: {audio_path}")
    
    try:
        # Load the audio file based on extension
        audio_load_start = time.time()
        file_extension = os.path.splitext(audio_path)[1].lower()
        
        if file_extension == '.m4a':
            audio = AudioSegment.from_file(audio_path, format="m4a")
            logger.info(f"Loaded M4A audio in {time.time() - audio_load_start:.2f} seconds")
        else:  # Default to MP3
            audio = AudioSegment.from_mp3(audio_path)
            logger.info(f"Loaded MP3 audio in {time.time() - audio_load_start:.2f} seconds")
        
        # Convert to mono (single channel)
        audio = audio.set_channels(1)
        
        # Convert to 16kHz sample rate
        audio = audio.set_frame_rate(16000)
        
        # Export as MP3 (convert M4A to MP3 if needed)
        export_start = time.time()
        mp3_path = audio_path
        if file_extension == '.m4a':
            mp3_path = audio_path.replace('.m4a', '.mp3')
            audio.export(mp3_path, format="mp3")
            logger.info(f"Converted M4A to MP3 and exported in {time.time() - export_start:.2f} seconds")
        else:
            audio.export(mp3_path, format="mp3")
            logger.info(f"Audio converted and exported in {time.time() - export_start:.2f} seconds")

        # Get model instance only when needed
        model_start = time.time()
        whisper_model = ASRModelSingleton.get_instance()
        logger.info(f"Model loaded in {time.time() - model_start:.2f} seconds")
        
        # Transcribe with faster_whisper
        transcribe_start = time.time()
        segments, info = whisper_model.transcribe(
            mp3_path
        )
        # log out info variable
        logger.info(f"Transcribe info: {info}")

        # Combine all segments into a single transcript
        transcript = "\n".join([f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}" for segment in segments])
        transcribe_duration = time.time() - transcribe_start
        
        total_duration = time.time() - start_time
        audio_duration = info.duration if hasattr(info, 'duration') else 0
        
        logger.info(f"Transcription completed in {transcribe_duration:.2f} seconds")
        logger.info(f"Total processing time: {total_duration:.2f} seconds for {audio_duration:.2f} seconds of audio")
        logger.info(f"Real-time factor: {transcribe_duration/audio_duration:.2f}x" if audio_duration > 0 else "Real-time factor: N/A")
        
        # Clean up temporary MP3 file if we converted from M4A
        if file_extension == '.m4a' and mp3_path != audio_path and os.path.exists(mp3_path):
            os.remove(mp3_path)
            logger.info(f"Cleaned up temporary MP3 file: {mp3_path}")
            
        return transcript
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}", exc_info=True)
        return "Transcription failed"
    finally:
        logger.info(f"Total time taken: {time.time() - start_time:.2f} seconds")