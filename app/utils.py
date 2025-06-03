import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile
import threading

# Singleton pattern for ASR model
class ASRModelSingleton:
    _instance = None
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    print("Loading ASR model...")
                    cls._instance = nemo_asr.models.ASRModel.from_pretrained(
                        model_name="nvidia/parakeet-tdt-0.6b-v2"
                    )
        return cls._instance

def convert_and_transcribe(mp3_path: str) -> str:
    wav_path = mp3_path.replace(".mp3", ".wav")
    
    # Load the audio file
    audio = AudioSegment.from_mp3(mp3_path)
    
    # Convert to mono (single channel)
    audio = audio.set_channels(1)
    
    # Convert to 16kHz sample rate
    audio = audio.set_frame_rate(16000)
    
    # Export as WAV
    audio.export(wav_path, format="wav")

    # Get model instance only when needed
    asr_model = ASRModelSingleton.get_instance()
    transcription = asr_model.transcribe([wav_path])[0].text
    os.remove(wav_path)
    return transcription