import os
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
import tempfile

# Load model once
asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")

def convert_and_transcribe(mp3_path: str) -> str:
    wav_path = mp3_path.replace(".mp3", ".wav")
    
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

    transcription = asr_model.transcribe([wav_path])[0]
    os.remove(wav_path)
    return transcription
