"""
Test script for MP3 and WAV audio compression in Kokoro TTS
"""
import numpy as np
import time
from entry.utils.audio import (
    audio_to_wav_bytes, 
    numpy_to_mp3_bytes,
    audio_to_bytes,
    audio_to_base64,
    create_audio_response,
    optimize_response_size
)
from loguru import logger
import io

# Configure logger
logger.add("test_mp3.log", rotation="1 MB")

def test_audio_formats():
    """Test different audio formats and quality levels"""
    print("Generating test audio data...")
    # Generate 30 seconds of test audio
    sample_rate = 24000
    duration_sec = 30
    audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, sample_rate * duration_sec))
    
    print(f"\nTest audio: {duration_sec}s at {sample_rate}Hz ({len(audio_data)} samples)")
    
    # Test WAV format at different quality levels
    print("\n=== WAV Format ===")
    for quality in ['high', 'medium', 'low']:
        start_time = time.time()
        wav_bytes = audio_to_wav_bytes(audio_data, sample_rate, quality)
        elapsed = time.time() - start_time
        print(f"Quality: {quality}")
        print(f"  - Size: {len(wav_bytes)/1024:.1f}KB")
        print(f"  - Encoding time: {elapsed:.3f}s")
    
    # Test MP3 format at different quality levels
    print("\n=== MP3 Format ===")
    for bitrate, quality in [('192k', 'high'), ('128k', 'medium'), ('64k', 'low')]:
        start_time = time.time()
        mp3_bytes = numpy_to_mp3_bytes(audio_data, sample_rate, bitrate)
        elapsed = time.time() - start_time
        print(f"Quality: {quality} ({bitrate})")
        print(f"  - Size: {len(mp3_bytes)/1024:.1f}KB")
        print(f"  - Encoding time: {elapsed:.3f}s")
    
    # Test optimization function
    print("\n=== Response Size Optimization ===")
    max_sizes = [1000, 500, 250, 100]  # KB
    
    for max_size in max_sizes:
        print(f"\nOptimizing for max size: {max_size}KB")
        
        # Test with WAV only
        buffer, quality, format = optimize_response_size(
            audio_data, sample_rate, max_size_kb=max_size, use_mp3=False
        )
        size = len(buffer.getvalue()) / 1024
        print(f"  WAV only: {quality} quality, {size:.1f}KB")
        
        # Test with MP3 allowed
        buffer, quality, format = optimize_response_size(
            audio_data, sample_rate, max_size_kb=max_size, use_mp3=True
        )
        size = len(buffer.getvalue()) / 1024
        print(f"  With MP3: {quality} quality {format}, {size:.1f}KB")

if __name__ == "__main__":
    print("Starting audio format tests...")
    test_audio_formats()
    print("\nTests completed!")
