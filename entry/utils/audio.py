"""
Audio processing utilities with compression options to reduce response size
"""
import io
import wave
import base64
import numpy as np
from typing import Tuple, Optional, Literal, Union
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# Audio format types
AudioFormat = Literal['wav', 'mp3']


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
        
        # Scale and convert to appropriate bit depth with dithering to reduce quantization noise
        scaled = np.clip(audio_data, -1.0, 1.0)
        if sampwidth == 1:  # 8-bit
            # Add small amount of dither noise before quantizing to reduce quantization artifacts
            dither = np.random.uniform(-0.5, 0.5, size=scaled.shape) / 127.0
            scaled = scaled + dither
            scaled = np.clip(scaled, -1.0, 1.0)  # Re-clip after adding dither
            scaled = (scaled * 127 + 128).astype(np.uint8)
        else:  # 16-bit
            # Add smaller amount of dither for 16-bit (less noticeable)
            dither = np.random.uniform(-0.5, 0.5, size=scaled.shape) / 32767.0
            scaled = scaled + dither
            scaled = np.clip(scaled, -1.0, 1.0)  # Re-clip after adding dither
            scaled = (scaled * 32767).astype(np.int16)
        
        wav_file.writeframes(scaled.tobytes())
    
    audio_buffer.seek(0)
    return audio_buffer.read()


def _downsample(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Downsample audio data to a lower sample rate using efficient resampling
    
    This uses a balance of quality and performance to maintain audio fidelity
    while keeping processing time reasonable.
    """
    # Calculate the resampling ratio
    ratio = original_rate / target_rate
    
    # For small ratio changes, use a simple but efficient method
    if ratio <= 3:
        # Apply a simple low-pass filter first to reduce aliasing
        # This is a basic FIR filter implementation
        if ratio > 1.5:  # Only apply filter if downsampling significantly
            # Simple moving average as a basic low-pass filter
            filter_size = min(int(ratio * 2), 10)  # Adaptive filter size
            filter_kernel = np.ones(filter_size) / filter_size
            # Apply the filter using numpy's convolve function
            audio_data = np.convolve(audio_data, filter_kernel, mode='same')
        
        # Then do the actual downsampling
        indices = np.arange(0, len(audio_data), ratio)
        indices = np.floor(indices).astype(int)
        indices = np.minimum(indices, len(audio_data) - 1)
        return audio_data[indices]
    
    # For larger ratio changes, use scipy if available for better quality
    try:
        from scipy import signal
        # Calculate the new length after resampling
        new_length = int(len(audio_data) / ratio)
        # Use the more efficient resample_poly instead of resample
        # This is much faster while still providing good quality
        gcd = np.gcd(original_rate, target_rate)
        up = target_rate // gcd
        down = original_rate // gcd
        resampled = signal.resample_poly(audio_data, up, down)
        # Trim to expected length (resample_poly might return slightly different length)
        if len(resampled) > new_length:
            resampled = resampled[:new_length]
        return resampled
    except (ImportError, AttributeError):
        # Fallback to a simple method if scipy isn't available
        logger.warning("scipy not available or incompatible, using simple downsampling")
        # Use a slightly better approach than just taking every nth sample
        indices = np.arange(0, len(audio_data), ratio)
        indices = np.floor(indices).astype(int)
        indices = np.minimum(indices, len(audio_data) - 1)
        return audio_data[indices]


def numpy_to_mp3_bytes(audio_data: np.ndarray, sample_rate: int = 24000, bitrate: str = '128k') -> bytes:
    """Convert numpy audio data to MP3 format bytes
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        bitrate: MP3 bitrate (e.g., '64k', '128k', '192k')
        
    Returns:
        MP3 encoded bytes
    """
    # First convert to WAV format (in memory)
    wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality='high')
    
    # Use pydub to convert WAV to MP3
    wav_audio = AudioSegment.from_wav(io.BytesIO(wav_bytes))
    
    # Export as MP3 to a bytes buffer
    mp3_buffer = io.BytesIO()
    wav_audio.export(mp3_buffer, format="mp3", bitrate=bitrate)
    mp3_buffer.seek(0)
    
    mp3_bytes = mp3_buffer.read()
    logger.info(f"MP3 encoded size ({bitrate}): {len(mp3_bytes)/1024:.1f}KB")
    return mp3_bytes


def audio_to_bytes(audio_data: np.ndarray, sample_rate: int = 24000, 
                  quality: str = 'high', format: AudioFormat = 'wav') -> bytes:
    """Convert audio data to bytes in the specified format with compression options
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
        format: Output format ('wav' or 'mp3')
        
    Returns:
        Audio bytes in the specified format
    """
    if format == 'mp3':
        # Map quality levels to MP3 bitrates
        bitrate_map = {
            'high': '192k',
            'medium': '128k',
            'low': '64k'
        }
        bitrate = bitrate_map.get(quality, '128k')
        return numpy_to_mp3_bytes(audio_data, sample_rate, bitrate)
    else:  # wav
        return audio_to_wav_bytes(audio_data, sample_rate, quality)


def audio_to_base64(audio_data: np.ndarray, sample_rate: int = 24000, 
                   quality: str = 'high', format: AudioFormat = 'wav') -> str:
    """Convert audio data to base64 encoded string with compression options
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
        format: Output format ('wav' or 'mp3')
    """
    audio_bytes = audio_to_bytes(audio_data, sample_rate, quality, format)
    encoded = base64.b64encode(audio_bytes).decode('utf-8')
    logger.info(f"Base64 encoded {format} size: {len(encoded)/1024:.1f}KB")
    return encoded


def create_audio_response(audio_data: np.ndarray, sample_rate: int = 24000, 
                       quality: str = 'medium', format: AudioFormat = 'wav') -> Tuple[io.BytesIO, str]:
    """Create an audio file response for streaming with compression options
    
    Args:
        audio_data: Audio data as numpy array
        sample_rate: Sample rate of the audio
        quality: Quality level ('high', 'medium', 'low')
            Default is 'medium' to balance quality and response size
        format: Output format ('wav' or 'mp3')
        
    Returns:
        Tuple of (BytesIO buffer, content_type)
    """
    audio_bytes = audio_to_bytes(audio_data, sample_rate, quality, format)
    buffer = io.BytesIO(audio_bytes)
    
    # Determine the content type based on format
    content_type = "audio/wav" if format == 'wav' else "audio/mpeg"
    
    logger.info(f"{format.upper()} response size ({quality}): {len(audio_bytes)/1024:.1f}KB")
    return buffer, content_type


# Keep the original function for backward compatibility
def create_wav_response(audio_data: np.ndarray, sample_rate: int = 24000, quality: str = 'medium') -> io.BytesIO:
    """Create a WAV file response for streaming with compression options (legacy function)"""
    buffer, _ = create_audio_response(audio_data, sample_rate, quality, format='wav')
    return buffer


def _try_audio_quality(audio_data: np.ndarray, sample_rate: int, quality: str, format: str) -> Tuple[bytes, str, str]:
    """Try to create audio with specified quality and format"""
    try:
        audio_bytes = audio_to_bytes(audio_data, sample_rate, quality, format)
        return audio_bytes, quality, format
    except Exception as e:
        logger.warning(f"Failed to create {quality} quality {format}: {str(e)}")
        return None, quality, format


def optimize_response_size(audio_data: np.ndarray, sample_rate: int = 24000,
                          max_size_kb: int = 1000, 
                          preferred_format: str = 'mp3') -> Tuple[io.BytesIO, str, str]:
    """
    Optimize audio response size by trying different quality/format combinations.
    Returns (audio_buffer, used_quality, used_format)
    """
    # Quality/format combinations to try, in order of preference
    combinations = [
        ('high', preferred_format),
        ('medium', preferred_format),
        ('low', preferred_format),
        ('low', 'wav' if preferred_format == 'mp3' else 'mp3'),  # Try alternative format
    ]
    
    for quality, format in combinations:
        audio_bytes, used_quality, used_format = _try_audio_quality(audio_data, sample_rate, quality, format)
        
        if audio_bytes is not None:
            size_kb = len(audio_bytes) / 1024
            if size_kb <= max_size_kb:
                logger.info(f"Optimized audio response using {quality} quality {format.upper()}. Size: {size_kb:.1f}KB")
                return io.BytesIO(audio_bytes), used_quality, used_format
            else:
                logger.warning(f"Audio response still exceeds {max_size_kb}KB with {quality} quality {format.upper()}")
    
    # If all combinations fail, use the last successful one (or fallback)
    logger.warning(f"Audio response still exceeds {max_size_kb}KB even with low quality MP3")
    fallback_bytes, _, _ = _try_audio_quality(audio_data, sample_rate, 'low', 'mp3')
    if fallback_bytes is None:
        # Ultimate fallback - use wav with low quality
        fallback_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality='low')
    
    logger.info(f"Using fallback audio response. Size: {len(fallback_bytes)/1024:.1f}KB")
    return io.BytesIO(fallback_bytes), 'low', 'mp3'