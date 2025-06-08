"""
Audio processing utilities
"""
import io
import wave
import base64
import numpy as np
from typing import Tuple


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert audio data to WAV format bytes"""
    audio_buffer = io.BytesIO()
    channels = 1
    sampwidth = 2
    
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        
        # Scale and convert to int16
        scaled = np.clip(audio_data, -1.0, 1.0)
        scaled = (scaled * 32767).astype(np.int16)
        wav_file.writeframes(scaled.tobytes())
    
    audio_buffer.seek(0)
    return audio_buffer.read()


def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000) -> str:
    """Convert audio data to base64 encoded WAV string"""
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate)
    return base64.b64encode(wav_bytes).decode('utf-8')


def create_wav_response(audio_data: np.ndarray, sample_rate: int = 24000) -> io.BytesIO:
    """Create a WAV file response for streaming"""
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate)
    return io.BytesIO(wav_bytes)