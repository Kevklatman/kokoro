"""
HLS streaming service for real-time TTS audio delivery
"""
import os
import io
import uuid
import asyncio
import numpy as np
from typing import Optional
from starlette.concurrency import run_in_threadpool

from entry.core.tts import select_voice_and_preset, preprocess_text, generate_audio


async def create_stream(
    text: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    use_gpu: bool = True,
    breathiness: float = 0.0,
    tenseness: float = 0.0,
    jitter: float = 0.0,
    sultry: float = 0.0,
    fiction: bool = False
) -> str:
    """Create a new HLS stream and return the stream ID"""
    stream_id = str(uuid.uuid4())
    stream_dir = f"streams/{stream_id}"
    os.makedirs(stream_dir, exist_ok=True)
    
    # Create initial playlist file
    playlist_path = f"{stream_dir}/playlist.m3u8"
    with open(playlist_path, "w") as f:
        f.write("#EXTM3U\n")
        f.write("#EXT-X-VERSION:3\n")
        f.write("#EXT-X-TARGETDURATION:2\n")
        f.write("#EXT-X-MEDIA-SEQUENCE:0\n")
    
    return stream_id


async def generate_streaming_audio(
    text: str, 
    stream_id: str,
    voice: Optional[str] = None,
    speed: float = 1.0,
    use_gpu: bool = True,
    breathiness: float = 0.0,
    tenseness: float = 0.0,
    jitter: float = 0.0,
    sultry: float = 0.0,
    fiction: bool = False
):
    """Generate audio in chunks and create HLS stream"""
    try:
        # Process text into sentences or paragraphs
        text_chunks = preprocess_text(text).split('\n\n')
        
        # Select voice based on fiction parameter if not specified
        voice_id, preset = select_voice_and_preset(voice, fiction=fiction)
        
        # Apply preset values if available
        if preset:
            if speed == 1.0 and "speed" in preset:
                speed = preset["speed"]
            if breathiness == 0.0 and "breathiness" in preset:
                breathiness = preset["breathiness"]
            if tenseness == 0.0 and "tenseness" in preset:
                tenseness = preset["tenseness"]
            if jitter == 0.0 and "jitter" in preset:
                jitter = preset["jitter"]
            if sultry == 0.0 and "sultry" in preset:
                sultry = preset["sultry"]
        
        # Create stream directory
        stream_dir = f"streams/{stream_id}"
        
        # Initialize HLS playlist
        segment_duration = 2  # seconds
        sample_rate = 24000
        
        # Process each chunk
        segment_index = 0
        for i, chunk in enumerate(text_chunks):
            if not chunk.strip():
                continue
                
            # Generate audio for this chunk
            (sample_rate, audio_data), _ = await run_in_threadpool(
                lambda: generate_audio(
                    chunk, 
                    voice_id, 
                    speed, 
                    use_gpu, 
                    breathiness, 
                    tenseness, 
                    jitter, 
                    sultry
                )
            )
            
            # Convert to int16 array
            scaled = np.clip(audio_data, -1.0, 1.0)
            audio_array = (scaled * 32767).astype(np.int16)
            
            # Split into segments of segment_duration
            samples_per_segment = int(segment_duration * sample_rate)
            for j in range(0, len(audio_array), samples_per_segment):
                segment = audio_array[j:j+samples_per_segment]
                
                # Pad last segment if needed
                if len(segment) < samples_per_segment:
                    segment = np.pad(segment, (0, samples_per_segment - len(segment)))
                
                # Save segment as TS file
                segment_file = f"{stream_dir}/segment_{segment_index}.ts"
                
                # Convert to bytes and save
                segment_bytes = segment.tobytes()
                with open(segment_file, "wb") as f:
                    f.write(segment_bytes)
                
                # Update playlist
                with open(f"{stream_dir}/playlist.m3u8", "a") as f:
                    f.write(f"#EXTINF:{segment_duration},\n")
                    f.write(f"segment_{segment_index}.ts\n")
                
                segment_index += 1
                
                # Small delay to simulate real-time generation
                await asyncio.sleep(0.1)
        
        # Mark the end of the playlist
        with open(f"{stream_dir}/playlist.m3u8", "a") as f:
            f.write("#EXT-X-ENDLIST\n")
            
        print(f"✅ Completed streaming audio generation for {stream_id}")
    except Exception as e:
        print(f"❌ Error generating streaming audio: {str(e)}")
        import traceback
        print(f"❌ Traceback: {traceback.format_exc()}")


def get_stream_file_path(stream_id: str, file_name: str) -> str:
    """Get the file path for a stream file"""
    return f"streams/{stream_id}/{file_name}"


def stream_file_exists(stream_id: str, file_name: str) -> bool:
    """Check if a stream file exists"""
    file_path = get_stream_file_path(stream_id, file_name)
    return os.path.exists(file_path)


def read_stream_file(stream_id: str, file_name: str) -> bytes:
    """Read a stream file and return its contents"""
    file_path = get_stream_file_path(stream_id, file_name)
    with open(file_path, "rb") as f:
        return f.read()


def read_stream_playlist(stream_id: str, file_name: str) -> str:
    """Read a stream playlist file and return its contents as text"""
    file_path = get_stream_file_path(stream_id, file_name)
    with open(file_path, "r") as f:
        return f.read()