"""
HLS streaming API routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Request
from fastapi.responses import Response

from entry.models import TTSRequest, StreamResponse
from entry.services.streaming import (
    create_stream, generate_streaming_audio, 
    stream_file_exists, read_stream_file, read_stream_playlist
)

router = APIRouter()


@router.post("/tts/stream", response_model=StreamResponse)
async def stream_tts(request: TTSRequest, background_tasks: BackgroundTasks, http_request: Request):
    """Stream audio as it's being generated using HLS"""
    try:
        # Create a new stream
        stream_id = await create_stream(
            request.text,
            request.voice,
            request.speed,
            request.use_gpu,
            request.breathiness,
            request.tenseness,
            request.jitter,
            request.sultry,
            request.fiction
        )
        
        # Start audio generation in background
        background_tasks.add_task(
            generate_streaming_audio,
            request.text,
            stream_id,
            request.voice,
            request.speed,
            request.use_gpu,
            request.breathiness,
            request.tenseness,
            request.jitter,
            request.sultry,
            request.fiction
        )
        
        # Return the stream URL
        base_url = str(http_request.base_url).rstrip('/')
        stream_url = f"{base_url}/streams/{stream_id}/playlist.m3u8"
        
        return StreamResponse(stream_url=stream_url, stream_id=stream_id)
        
    except Exception as e:
        print(f"❌ Streaming error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{stream_id}/{file_name}")
async def get_stream_file(stream_id: str, file_name: str):
    """Serve HLS stream files"""
    if not stream_file_exists(stream_id, file_name):
        raise HTTPException(status_code=404, detail="Stream file not found")
    
    # For m3u8 playlists
    if file_name.endswith(".m3u8"):
        content = read_stream_playlist(stream_id, file_name)
        return Response(content=content, media_type="application/vnd.apple.mpegurl")
    
    # For TS segments
    elif file_name.endswith(".ts"):
        content = read_stream_file(stream_id, file_name)
        return Response(content=content, media_type="video/mp2t")
    
    raise HTTPException(status_code=400, detail="Invalid stream file type")