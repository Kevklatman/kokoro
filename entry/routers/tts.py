"""
TTS API routes
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import base64

from entry.models import (
    TTSRequest, TTSBatchRequest, TokenizeRequest, PreprocessRequest,
    TTSResponse, BatchTTSResponse, TokenizeResponse, PreprocessResponse
)
from entry.core.tts import (
    generate_audio, generate_audio_batch, tokenize_text, 
    preprocess_text, select_voice_and_preset
)
from entry.utils.audio import audio_to_base64, create_wav_response, create_audio_response, optimize_response_size, audio_to_bytes
from loguru import logger

router = APIRouter()


@router.post("/", response_class=StreamingResponse)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech and return audio"""
    try:
        # Extract request parameters
        text = request.text
        voice = request.voice
        speed = request.speed
        quality = getattr(request, 'quality', 'auto') if hasattr(request, 'quality') else 'auto'
        format = getattr(request, 'format', 'auto') if hasattr(request, 'format') else 'auto'
        
        logger.info(f"TTS request for voice {voice} with quality {quality} and format {format}")
        
        # Select voice and preset
        preset_name = getattr(request, 'preset', None) if hasattr(request, 'preset') else None
        selected_voice, emotion_preset = select_voice_and_preset(
            request.voice, preset_name, fiction=request.fiction
        )

        # Apply preset values if available (ENFORCE presets)
        if emotion_preset:
            speed = emotion_preset['speed']
            breathiness = emotion_preset['breathiness']
            tenseness = emotion_preset['tenseness']
            jitter = emotion_preset['jitter']
            sultry = emotion_preset['sultry']
        else:
            breathiness = request.breathiness
            tenseness = request.tenseness
            jitter = request.jitter
            sultry = request.sultry

        preprocessed_text = preprocess_text(request.text)

        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text,
            selected_voice,
            speed,
            request.use_gpu,
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Audio generation failed")

        # Handle different quality and format settings
        if quality == 'auto' or format == 'auto':
            # Automatically optimize response size to avoid Cloud Run limits
            audio_buffer, used_quality, used_format = optimize_response_size(
                audio_data, sample_rate, max_size_kb=30000, use_mp3=(format != 'wav')
            )
            logger.info(f"Auto-selected quality: {used_quality}, format: {used_format} for response")
            content_type = "audio/mpeg" if used_format == 'mp3' else "audio/wav"
            file_ext = "mp3" if used_format == 'mp3' else "wav"
        else:
            # Use the specified quality and format
            audio_buffer, content_type = create_audio_response(audio_data, sample_rate, quality=quality, format=format)
            file_ext = "mp3" if format == 'mp3' else "wav"
            
        return StreamingResponse(
            audio_buffer, 
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=tts_{selected_voice}_{quality}.{file_ext}"
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchTTSResponse)
async def batch_text_to_speech(request: TTSBatchRequest):
    """Convert a batch of texts to speech and return base64-encoded audio strings"""
    try:
        fiction_list = request.fiction if request.fiction else [False] * len(request.texts)
        
        results = []
        for idx, text in enumerate(request.texts):
            is_fiction = fiction_list[idx] if idx < len(fiction_list) else False
            
            # Apply presets based on fiction flag
            if is_fiction:
                # Bella preset for literature
                voice = 'af_heart'
                speed = 1.1
                breathiness = 0.1
                tenseness = 0.1
                jitter = 0.15
                sultry = 0.1
            else:
                # Sky preset for articles
                voice = 'af_sky'
                speed = 1.0
                breathiness = 0.15
                tenseness = 0.5
                jitter = 0.3
                sultry = 0.1
            
            (sample_rate, audio_data), _ = generate_audio(
                text, voice, speed, request.use_gpu,
                breathiness, tenseness, jitter, sultry
            )
            results.append((sample_rate, audio_data))
        
        # Get quality parameter with default of 'auto'
        # Apply quality and format settings
        quality = request.quality.value if hasattr(request, 'quality') else 'medium'
        format = request.format.value if hasattr(request, 'format') else 'wav'
        logger.info(f"Batch TTS request with quality {quality}, format {format}")
        
        audio_base64_list = []
        for sample_rate, audio_data in results:
            if quality == 'auto':
                # For batch requests, use medium quality by default
                if format == 'auto':
                    # Default to MP3 for batch requests to save bandwidth
                    audio_base64 = audio_to_base64(audio_data, sample_rate, quality='medium', format='mp3')
                else:
                    # Use the specified format
                    audio_base64 = audio_to_base64(audio_data, sample_rate, quality='medium', format=format)
            else:
                # Use the specified quality and format
                if format == 'auto':
                    # Default to MP3 for batch requests
                    audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format='mp3')
                else:
                    audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format=format)
            audio_base64_list.append(audio_base64)
        
        return BatchTTSResponse(audios=audio_base64_list)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/base64", response_model=TTSResponse)
async def text_to_speech_base64(request: TTSRequest):
    """Convert text to speech and return audio as base64 encoded string"""
    try:
        selected_voice, emotion_preset = select_voice_and_preset(
            request.voice, fiction=request.fiction
        )
        
        # Apply preset values or use request values
        speed = request.speed
        breathiness = request.breathiness
        tenseness = request.tenseness
        jitter = request.jitter
        sultry = request.sultry
        
        if emotion_preset:
            if request.speed == 1.0 and "speed" in emotion_preset:
                speed = emotion_preset["speed"]
            if request.breathiness == 0.0 and "breathiness" in emotion_preset:
                breathiness = emotion_preset["breathiness"]
            if request.tenseness == 0.0 and "tenseness" in emotion_preset:
                tenseness = emotion_preset["tenseness"]
            if request.jitter == 0.0 and "jitter" in emotion_preset:
                jitter = emotion_preset["jitter"]
            if request.sultry == 0.0 and "sultry" in emotion_preset:
                sultry = emotion_preset["sultry"]

        preprocessed_text = preprocess_text(request.text)
        
        (sample_rate, audio_data), phonemes = generate_audio(
            preprocessed_text, 
            selected_voice,
            speed, 
            request.use_gpu,
            breathiness,
            tenseness,
            jitter,
            sultry
        )
        
        if audio_data is None:
            raise HTTPException(status_code=500, detail="Failed to generate audio")
        
        # Get quality and format parameters
        quality = request.quality.value
        format = request.format.value if hasattr(request, 'format') else 'auto'
        logger.info(f"Base64 TTS request with quality {quality}, format {format}")
        
        if quality == 'auto' or format == 'auto':
            # For base64 responses, optimize size based on quality and format
            audio_buffer, used_quality, used_format = optimize_response_size(
                audio_data, sample_rate, max_size_kb=30000, use_mp3=(format != 'wav')
            )
            # Convert the optimized buffer to base64
            audio_bytes = audio_buffer.getvalue()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
            logger.info(f"Auto-selected quality: {used_quality}, format: {used_format} for base64 response")
            # Remember the format for the response
            format = used_format
        else:
            # Use the specified quality and format
            audio_base64 = audio_to_base64(audio_data, sample_rate, quality=quality, format=format)
        
        # Get the actual sample rate after potential downsampling
        actual_sample_rate = sample_rate
        if quality == 'medium' or (quality == 'auto' and used_quality == 'medium'):
            actual_sample_rate = 16000
        elif quality == 'low' or (quality == 'auto' and used_quality == 'low'):
            actual_sample_rate = 8000
        
        return TTSResponse(
            sample_rate=actual_sample_rate,
            audio_base64=audio_base64,
            phonemes=phonemes
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """Tokenize text to phonemes without generating audio"""
    try:
        phonemes = tokenize_text(request.text, request.voice)
        return TokenizeResponse(phonemes=phonemes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess_text_endpoint(request: PreprocessRequest):
    """Preprocess text for TTS"""
    processed = preprocess_text(request.text)
    return PreprocessResponse(original=request.text, processed=processed)