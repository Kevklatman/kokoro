"""
Audio processing utilities with compression options to reduce response size
"""
import io
import wave
import base64
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int = 24000, quality: str = 'high') -> bytes:
    """Convert audio data to WAV format bytes with optional quality reduction
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
            - high: 24kHz, 16-bit
            - medium: 16kHz, 8-bit (reduced bit depth)
            - low: 8kHz, 8-bit
    """
    audio_buffer = io.BytesIO()
    channels = 1
    
    # Adjust sample rate and bit depth based on quality
    if quality == 'low':
        target_sample_rate = 8000
        sampwidth = 1  # 8-bit
        # Downsample if needed
        if sample_rate != target_sample_rate:
            audio_data = _downsample(audio_data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    elif quality == 'medium':
        target_sample_rate = 16000
        sampwidth = 1  # 8-bit (reduced from 16-bit)
        # Downsample if needed
        if sample_rate != target_sample_rate:
            audio_data = _downsample(audio_data, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    else:  # high quality
        sampwidth = 2  # 16-bit
    
    # Log the size reduction
    original_size = len(audio_data) * (2 if quality == 'high' else (2 if quality == 'medium' else 1))
    logger.info(f"Audio quality: {quality}, estimated size: {original_size/1024:.1f}KB")
    
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        
        # Scale and convert to appropriate bit depth
        scaled = np.clip(audio_data, -1.0, 1.0)
        if sampwidth == 1:  # 8-bit
            scaled = (scaled * 127 + 128).astype(np.uint8)
        else:  # 16-bit
            scaled = (scaled * 32767).astype(np.int16)
        
        wav_file.writeframes(scaled.tobytes())
    
    audio_buffer.seek(0)
    return audio_buffer.read()


def _downsample(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Downsample audio data to a lower sample rate"""
    # Simple downsampling by taking every nth sample
    ratio = original_rate // target_rate
    return audio_data[::ratio]


def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000, quality: str = 'high') -> str:
    """Convert audio data to base64 encoded WAV string with compression options
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
    """
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
    encoded = base64.b64encode(wav_bytes).decode('utf-8')
    logger.info(f"Base64 encoded audio size: {len(encoded)/1024:.1f}KB")
    return encoded


def create_wav_response(audio_data: np.ndarray, sample_rate: int = 24000, quality: str = 'medium') -> io.BytesIO:
    """Create a WAV file response for streaming with compression options
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
            Default is 'medium' to balance quality and response size
    """
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
    buffer = io.BytesIO(wav_bytes)
    logger.info(f"WAV response size: {len(wav_bytes)/1024:.1f}KB")
    return buffer


def optimize_response_size(audio_data: np.ndarray, sample_rate: int = 24000, max_size_kb: int = 10000) -> Tuple[io.BytesIO, str]:
    """Automatically optimize audio response size to fit within Cloud Run limits
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        max_size_kb: Maximum allowed response size in KB (default: 10MB)
        
    Returns:
        Tuple of (BytesIO buffer, quality level used)
    """
    # Try high quality first
    quality = 'high'
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
    
    # If too large, try medium quality
    if len(wav_bytes) > max_size_kb * 1024:
        quality = 'medium'
        wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
        
        # If still too large, use low quality
        if len(wav_bytes) > max_size_kb * 1024:
            quality = 'low'
            wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
            
            # If still too large, log a warning
            if len(wav_bytes) > max_size_kb * 1024:
                logger.warning(f"Audio response still exceeds {max_size_kb}KB even at lowest quality")
    
    logger.info(f"Optimized audio response using {quality} quality. Size: {len(wav_bytes)/1024:.1f}KB")
    return io.BytesIO(wav_bytes), quality