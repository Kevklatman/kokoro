"""
Audio processing utilities with compression options to reduce response size
"""
import io
import wave
import base64
import numpy as np
from typing import Tuple, Optional, Literal, Union
from loguru import logger
from pydub import AudioSegment
from entry.utils.string_utils import format_size_info, normalize_string_case

# Audio format types
AudioFormat = Literal['wav', 'mp3']


def ensure_audio_array(audio_data, fallback_length: int = 1000) -> np.ndarray:
    """Ensure audio_data is a numpy array, with fallback to silent audio if conversion fails"""
    if isinstance(audio_data, np.ndarray):
        return audio_data
    
    try:
        return np.array(audio_data, dtype=np.float32)
    except Exception:
        return np.zeros(fallback_length, dtype=np.float32)


def normalize_audio_for_wav(audio_data: np.ndarray, bit_depth: int = 16) -> np.ndarray:
    """Normalize audio data for WAV format with specified bit depth"""
    # Clip to valid range
    scaled = np.clip(audio_data, -1.0, 1.0)
    
    if bit_depth == 8:
        # Add dither for 8-bit
        dither = np.random.uniform(-0.5, 0.5, size=scaled.shape) / 127.0
        scaled = scaled + dither
        scaled = np.clip(scaled, -1.0, 1.0)  # Re-clip after adding dither
        return (scaled * 127 + 128).astype(np.uint8)
    else:  # 16-bit
        # Add dither for 16-bit
        dither = np.random.uniform(-0.5, 0.5, size=scaled.shape) / 32767.0
        scaled = scaled + dither
        scaled = np.clip(scaled, -1.0, 1.0)  # Re-clip after adding dither
        return (scaled * 32767).astype(np.int16)


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
    
    # Ensure audio_data is a numpy array
    audio_data = ensure_audio_array(audio_data)
    
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
    
    # Normalize audio for WAV format
    audio_array = normalize_audio_for_wav(audio_data, bit_depth=8 if sampwidth == 1 else 16)
    
    # Write WAV file
    with wave.open(audio_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())
    
    return audio_buffer.getvalue()


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


def encode_audio_base64(audio_bytes: bytes) -> str:
    """Encode audio bytes to base64 string"""
    return base64.b64encode(audio_bytes).decode('utf-8')


def format_audio_size(audio_bytes: bytes, format_name: str = "audio") -> str:
    """Format audio size in KB with format name"""
    size_kb = len(audio_bytes) / 1024
    return format_size_info(format_name, size_kb)


def format_quality_info(format_name: str, quality: str, audio_bytes: bytes) -> str:
    """Format quality information for logging"""
    size_kb = len(audio_bytes) / 1024
    return format_size_info(format_name, size_kb, quality)


def normalize_audio_data(audio_data: np.ndarray, target_sample_rate: int = 24000) -> np.ndarray:
    """Normalize audio data to target sample rate and ensure proper format"""
    if audio_data is None:
        raise ValueError("Audio data is None")
    
    if len(audio_data) == 0:
        raise ValueError("Audio data is empty")
    
    # Ensure audio is 1D
    if audio_data.ndim > 1:
        audio_data = audio_data.flatten()
    
    # Normalize to float32 if needed
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Ensure values are in valid range
    if np.max(np.abs(audio_data)) > 1.0:
        audio_data = np.clip(audio_data, -1.0, 1.0)
    
    return audio_data


def validate_audio_format(format_name: str) -> str:
    """Validate and normalize audio format name"""
    valid_formats = ['wav', 'mp3', 'flac', 'ogg', 'auto']
    format_lower = normalize_string_case(format_name, "lower")
    
    if format_lower not in valid_formats:
        raise ValueError(f"Unsupported audio format: {format_name}. Supported: {valid_formats}")
    
    # Map 'auto' to 'wav' as default
    if format_lower == 'auto':
        return 'wav'
    
    return format_lower


def validate_audio_quality(quality: str) -> str:
    """Validate and normalize audio quality setting"""
    valid_qualities = ['low', 'medium', 'high', 'auto']
    quality_lower = normalize_string_case(quality, "lower")
    
    if quality_lower not in valid_qualities:
        raise ValueError(f"Unsupported audio quality: {quality}. Supported: {valid_qualities}")
    
    # Map 'auto' to 'high' as default
    if quality_lower == 'auto':
        return 'high'
    
    return quality_lower


def get_audio_quality_settings(quality: str) -> dict:
    """Get audio quality settings based on quality level"""
    quality_settings = {
        'low': {
            'sample_rate': 16000,
            'bitrate': '64k',
            'channels': 1
        },
        'medium': {
            'sample_rate': 22050,
            'bitrate': '128k',
            'channels': 1
        },
        'high': {
            'sample_rate': 24000,
            'bitrate': '192k',
            'channels': 1
        }
    }
    
    normalized_quality = validate_audio_quality(quality)
    return quality_settings[normalized_quality]


def resample_audio(audio_data: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    """Resample audio to target sample rate"""
    if original_rate == target_rate:
        return audio_data
    
    ratio = original_rate / target_rate
    indices = np.arange(0, len(audio_data), ratio)
    indices = np.minimum(indices, len(audio_data) - 1)
    
    return audio_data[indices.astype(int)]


def optimize_audio_size(audio_data: np.ndarray, target_size_kb: int, format_name: str = 'wav') -> tuple[bytes, str]:
    """Optimize audio size by reducing quality until target size is met"""
    format_lower = validate_audio_format(format_name)
    
    # Try different quality levels
    for quality in ['high', 'medium', 'low']:
        try:
            quality_settings = get_audio_quality_settings(quality)
            
            if format_lower == 'mp3':
                audio_bytes = encode_mp3(audio_data, quality_settings['bitrate'])
            else:
                audio_bytes = encode_wav(audio_data, quality_settings['sample_rate'])
            
            size_kb = len(audio_bytes) / 1024
            
            if size_kb <= target_size_kb:
                logger.info(f"Optimized audio response using {quality} quality {format_lower.upper()}. Size: {size_kb:.1f}KB")
                return audio_bytes, quality
            
        except Exception as e:
            logger.warning(f"Failed to encode with {quality} quality: {e}")
            continue
    
    # Fallback to lowest quality
    logger.warning(f"Audio response still exceeds {target_size_kb}KB with {quality} quality {format_lower.upper()}")
    return encode_wav(audio_data, 16000), 'low'


def encode_audio_to_format(audio_data: np.ndarray, format_name: str, quality: str = 'high') -> bytes:
    """Encode audio data to specified format with quality settings"""
    format_lower = validate_audio_format(format_name)
    quality_settings = get_audio_quality_settings(quality)
    
    if format_lower == 'mp3':
        return encode_mp3(audio_data, quality_settings['bitrate'])
    elif format_lower == 'wav':
        return encode_wav(audio_data, quality_settings['sample_rate'])
    else:
        # For other formats, use wav as fallback
        return encode_wav(audio_data, quality_settings['sample_rate'])


def create_audio_response(audio_data: np.ndarray, format_name: str, quality: str, max_size_kb: int = 1024) -> dict:
    """Create a complete audio response with optimization"""
    # Normalize audio data
    audio_data = normalize_audio_data(audio_data)
    
    # Encode audio
    audio_bytes = encode_audio_to_format(audio_data, format_name, quality)
    
    # Check if size optimization is needed
    size_kb = len(audio_bytes) / 1024
    if size_kb > max_size_kb:
        audio_bytes, actual_quality = optimize_audio_size(audio_data, max_size_kb, format_name)
        quality = actual_quality
    
    # Encode to base64 if needed
    audio_base64 = encode_audio_base64(audio_bytes)
    
    return {
        'audio': audio_base64,
        'format': format_name,
        'quality': quality,
        'size_kb': len(audio_bytes) / 1024
    }


def encode_mp3(audio_data: np.ndarray, bitrate: str = '192k') -> bytes:
    """Encode audio data to MP3 format"""
    # Ensure audio is in the correct format for MP3 encoding
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Convert bitrate string to integer
    bitrate_int = int(bitrate.replace('k', '000'))
    
    # Encode to MP3
    mp3_bytes = io.BytesIO()
    sf.write(mp3_bytes, audio_data, 24000, format='mp3', subtype='MPEG_LAYER_III', bitrate=bitrate_int)
    mp3_bytes.seek(0)
    
    encoded_bytes = mp3_bytes.read()
    logger.info(f"MP3 encoded size ({bitrate}): {len(encoded_bytes)/1024:.1f}KB")
    
    return encoded_bytes


def encode_wav(audio_data: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Encode audio data to WAV format"""
    # Ensure audio is in the correct format
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # Encode to WAV
    wav_bytes = io.BytesIO()
    sf.write(wav_bytes, audio_data, sample_rate, format='wav', subtype='PCM_16')
    wav_bytes.seek(0)
    
    return wav_bytes.read()


def create_fallback_response(audio_data: np.ndarray) -> dict:
    """Create a fallback audio response with minimal quality"""
    try:
        # Use lowest quality settings
        fallback_bytes = encode_wav(audio_data, 16000)
        logger.info(f"Using fallback audio response. Size: {len(fallback_bytes)/1024:.1f}KB")
        
        return {
            'audio': encode_audio_base64(fallback_bytes),
            'format': 'wav',
            'quality': 'low',
            'size_kb': len(fallback_bytes) / 1024,
            'warning': 'Audio optimized to minimum quality due to size constraints'
        }
    except Exception as e:
        logger.error(f"Failed to create fallback response: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate audio response")


def audio_to_base64(audio_data: np.ndarray, format: str = 'wav', quality: str = 'high') -> str:
    """Convert audio data to base64 string"""
    format_lower = normalize_string_case(format, "lower")
    
    if format_lower == 'mp3':
        audio_bytes = encode_mp3(audio_data, '192k' if quality == 'high' else '128k' if quality == 'medium' else '64k')
    else:
        audio_bytes = encode_wav(audio_data, 24000 if quality == 'high' else 22050 if quality == 'medium' else 16000)
    
    encoded = encode_audio_base64(audio_bytes)
    logger.info(f"Base64 encoded {format} size: {len(encoded)/1024:.1f}KB")
    return encoded


def optimize_audio_response(audio_data: np.ndarray, format: str, quality: str, max_size_kb: int = 1024) -> Tuple[bytes, str]:
    """Optimize audio response to fit within size constraints"""
    original_size = len(audio_data) * (2 if quality == 'high' else (2 if quality == 'medium' else 1))
    
    # Try different quality levels
    for test_quality in ['high', 'medium', 'low']:
        try:
            format_lower = normalize_string_case(format, "lower")
            
            if format_lower == 'mp3':
                bitrate = '192k' if test_quality == 'high' else '128k' if test_quality == 'medium' else '64k'
                audio_bytes = encode_mp3(audio_data, bitrate)
            else:
                sample_rate = 24000 if test_quality == 'high' else 22050 if test_quality == 'medium' else 16000
                audio_bytes = encode_wav(audio_data, sample_rate)
            
            size_kb = len(audio_bytes) / 1024
            if size_kb <= max_size_kb:
                logger.info(f"Optimized audio response using {test_quality} quality {format.upper()}. Size: {size_kb:.1f}KB")
                return audio_bytes, test_quality
                
        except Exception as e:
            logger.warning(f"Failed to encode with {test_quality} quality: {e}")
            continue
    
    # Fallback to lowest quality
    logger.warning(f"Audio response still exceeds {max_size_kb}KB with {quality} quality {format.upper()}")
    return encode_wav(audio_data, 16000), 'low'


def resample_audio_quality(audio_data: np.ndarray, quality: str) -> np.ndarray:
    """Resample audio based on quality setting"""
    if quality == 'high':
        return audio_data  # Keep original sample rate
    elif quality == 'medium':
        ratio = 24000 / 22050
        indices = np.arange(0, len(audio_data), ratio)
        indices = np.minimum(indices, len(audio_data) - 1)
        resampled = audio_data[indices.astype(int)]
        
        # Ensure we don't exceed original length
        new_length = int(len(audio_data) / ratio)
        if len(resampled) > new_length:
            resampled = resampled[:new_length]
        
        return resampled
    else:  # low quality
        ratio = 24000 / 16000
        indices = np.arange(0, len(audio_data), ratio)
        indices = np.minimum(indices, len(audio_data) - 1)
        resampled = audio_data[indices.astype(int)]
        
        # Ensure we don't exceed original length
        new_length = int(len(audio_data) / ratio)
        if len(resampled) > new_length:
            resampled = resampled[:new_length]
        
        return resampled