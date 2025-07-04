"""
TTS API routes
"""
from fastapi import APIRouter, HTTPException, Response, Request
from fastapi.responses import StreamingResponse, JSONResponse
import base64
import numpy as np

from entry.models import (
    TTSRequest, TTSBatchRequest, TokenizeRequest, PreprocessRequest,
    TTSResponse, BatchTTSResponse, TokenizeResponse, PreprocessResponse
)
from entry.core.tts import (
    generate_audio, generate_audio_batch, tokenize_text, 
    preprocess_text, select_voice_and_preset, is_gpu_available, get_model_components
)
from entry.core.models import get_settings
from entry.utils.audio import (
    audio_to_base64, create_audio_response, 
    audio_to_bytes, ensure_audio_array,
    encode_audio_base64, format_quality_info, normalize_audio_data,
    validate_audio_format, validate_audio_quality, encode_audio_to_format,
    optimize_audio_size
)
from entry.utils.error_handling import (
    safe_execute, handle_http_error, create_validation_error,
    extract_emotion_params, log_operation_start, log_operation_success
)
from entry.utils.batch_processing import (
    categorize_items, process_batch_items, merge_batch_results,
    validate_batch_request, create_batch_response
)
from entry.utils.dict_utils import extract_dict_values, safe_dict_get
from entry.utils.list_utils import extend_list_to_length, safe_list_append
from loguru import logger

router = APIRouter()


@router.post("/", response_class=Response)
async def text_to_speech(request: TTSRequest):
    """Convert text to speech"""
    def generate_audio_safe():
        # Get model components
        models, pipelines, voices = get_model_components()
        
        # Validate request parameters
        voice = request.voice or 'af_sky'
        speed = max(0.1, min(5.0, request.speed or 1.0))
        use_gpu = is_gpu_available() if request.use_gpu is None else request.use_gpu
        
        # Select voice and preset
        selected_voice, emotion_preset = select_voice_and_preset(
            voice, request.preset_name, request.fiction
        )
        
        # Extract emotion parameters
        emotion_params = extract_emotion_params(emotion_preset)
        
        # Generate audio
        sample_rate, audio_data = generate_audio(
            text=request.text,
            voice=selected_voice,
            speed=speed,
            use_gpu=use_gpu,
            **emotion_params
        )
        
        # Create audio response
        format_name = validate_audio_format(request.format or 'wav')
        quality = validate_audio_quality(request.quality or 'high')
        
        return create_audio_response(
            audio_data=audio_data,
            format_name=format_name,
            quality=quality,
            max_size_kb=request.max_size_kb or 1024
        )
    
    log_operation_start("TTS generation", voice=request.voice, text_length=len(request.text))
    
    try:
        result = safe_execute(generate_audio_safe, context="TTS generation")
        log_operation_success("TTS generation", voice=request.voice)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise handle_http_error(e, context="TTS generation")


@router.post("/batch")
async def batch_text_to_speech(request: Request):
    """Convert multiple texts to speech in batch"""
    try:
        request_data = await request.json()
        
        # Validate request
        texts = request_data.get("texts", [])
        if not texts:
            raise create_validation_error("No texts provided")
        
        # Validate batch request
        is_valid, error_msg = validate_batch_request(
            texts=texts,
            required_params={"voice": request_data.get("voice")}
        )
        if not is_valid:
            raise create_validation_error(error_msg)
        
        # Extract parameters using dictionary utilities
        param_keys = ["voice", "speed", "breathiness", "tenseness", "jitter", "sultry", "quality", "format"]
        param_defaults = ["af_sky", 1.0, 0.0, 0.0, 0.0, 0.0, "medium", "mp3"]
        params = extract_dict_values(request_data, param_keys, param_defaults)
        
        voice = params["voice"]
        speed = params["speed"]
        breathiness = params["breathiness"]
        tenseness = params["tenseness"]
        jitter = params["jitter"]
        sultry = params["sultry"]
        quality = params["quality"]
        format_type = params["format"]
        
        fiction_list = safe_dict_get(request_data, "fiction", [False] * len(texts))
        
        # Ensure fiction_list matches texts length
        extend_list_to_length(fiction_list, len(texts), False)
        
        log_operation_start("batch TTS", count=len(texts), voice=voice)
        
        # Categorize texts by fiction/non-fiction
        def categorize_by_fiction(item):
            idx, text = item
            return "fiction" if fiction_list[idx] else "nonfiction"
        
        categorized = categorize_items(
            items=list(enumerate(texts)),
            categories=["fiction", "nonfiction"],
            category_func=categorize_by_fiction
        )
        
        # Process fiction texts
        fiction_results = []
        if categorized["fiction"]:
            fiction_indices, fiction_texts = zip(*categorized["fiction"])
            fiction_results = process_batch_items(
                items=fiction_texts,
                processor_func=lambda text: generate_audio(
                    text=text, voice=voice, speed=speed,
                    breathiness=breathiness, tenseness=tenseness,
                    jitter=jitter, sultry=sultry
                ),
                batch_name="fiction texts"
            )
        
        # Process non-fiction texts
        nonfiction_results = []
        if categorized["nonfiction"]:
            nonfiction_indices, nonfiction_texts = zip(*categorized["nonfiction"])
            nonfiction_results = process_batch_items(
                items=nonfiction_texts,
                processor_func=lambda text: generate_audio(
                    text=text, voice=voice, speed=speed,
                    breathiness=breathiness, tenseness=tenseness,
                    jitter=jitter, sultry=sultry
                ),
                batch_name="non-fiction texts"
            )
        
        # Merge results back to original order
        all_results = [None] * len(texts)
        
        # Merge fiction results
        if categorized["fiction"]:
            fiction_indices, _ = zip(*categorized["fiction"])
            for idx, result in zip(fiction_indices, fiction_results):
                if idx < len(all_results) and result is not None:
                    all_results[idx] = result
        
        # Merge non-fiction results
        if categorized["nonfiction"]:
            nonfiction_indices, _ = zip(*categorized["nonfiction"])
            for idx, result in zip(nonfiction_indices, nonfiction_results):
                if idx < len(all_results) and result is not None:
                    all_results[idx] = result
        
        # Convert results to base64
        audio_base64_list = []
        for i, result in enumerate(all_results):
            if result is None:
                safe_list_append(audio_base64_list, "")
                continue
            
            try:
                sample_rate, audio_data = result
                encoded_audio = audio_to_base64(audio_data, format_type, quality)
                safe_list_append(audio_base64_list, encoded_audio)
            except Exception as e:
                logger.error(f"Error encoding audio for index {i}: {str(e)}")
                safe_list_append(audio_base64_list, "")
        
        response = create_batch_response(
            results=audio_base64_list,
            format_type=format_type,
            include_metadata=True
        )
        
        log_operation_success("batch TTS", count=len(texts), success_count=response["success_count"])
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise handle_http_error(e, context="batch TTS")


@router.post("/base64", response_model=TTSResponse)
async def text_to_speech_base64(request: TTSRequest):
    """Convert text to speech and return as base64"""
    def generate_base64_safe():
        # Get model components
        models, pipelines, voices = get_model_components()
        
        # Validate request parameters
        voice = request.voice or 'af_sky'
        speed = max(0.1, min(5.0, request.speed or 1.0))
        use_gpu = is_gpu_available() if request.use_gpu is None else request.use_gpu
        
        # Select voice and preset
        selected_voice, emotion_preset = select_voice_and_preset(
            voice, request.preset_name, request.fiction
        )
        
        # Extract emotion parameters
        emotion_params = extract_emotion_params(emotion_preset)
        
        # Generate audio
        sample_rate, audio_data = generate_audio(
            text=request.text,
            voice=selected_voice,
            speed=speed,
            use_gpu=use_gpu,
            **emotion_params
        )
        
        # Encode to base64
        format_name = validate_audio_format(request.format or 'wav')
        quality = validate_audio_quality(request.quality or 'high')
        
        audio_bytes = encode_audio_to_format(audio_data, format_name, quality)
        audio_base64 = encode_audio_base64(audio_bytes)
        
        return {
            "audio": audio_base64,
            "format": format_name,
            "sample_rate": sample_rate,
            "size_kb": len(audio_bytes) / 1024
        }
    
    log_operation_start("TTS base64 generation", voice=request.voice, text_length=len(request.text))
    
    try:
        result = safe_execute(generate_base64_safe, context="TTS base64 generation")
        log_operation_success("TTS base64 generation", voice=request.voice)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise handle_http_error(e, context="TTS base64 generation")


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