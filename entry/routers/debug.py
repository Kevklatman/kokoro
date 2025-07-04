"""
Debug router for TTS functionality
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from loguru import logger
from typing import Dict, Any, Optional

from entry.core.tts import preprocess_text, generate_audio, select_voice_and_preset, is_gpu_available, get_gpu_settings, get_model_components, tokenize_text
from entry.config import get_settings, parse_bool_env
import torch
from entry.utils.error_handling import (
    safe_execute, handle_http_error, log_operation_start, log_operation_success, log_operation_failure
)
from entry.utils.dict_utils import safe_dict_update

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
)

class DebugTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    fiction: bool = False
    speed: float = 1.0
    breathiness: float = 0.0
    tenseness: float = 0.0
    jitter: float = 0.0
    sultry: float = 0.0
    use_gpu: bool = False

class DebugResponse(BaseModel):
    success: bool
    error: Optional[str] = None
    audio_length: int
    sample_rate: float
    phoneme_length: int
    system_info: Dict[str, Any]
    phonemes: str

@router.post("/tts-pipeline", response_model=DebugResponse)
async def debug_tts_pipeline(request: DebugTTSRequest) -> DebugResponse:
    """Debug TTS pipeline with detailed information"""
    log_operation_start("TTS pipeline debug", voice=request.voice, text_length=len(request.text))
    
    def debug_pipeline_safe():
        # Get model components
        models, pipelines, voices = get_model_components()
        
        # Get GPU settings
        cuda_available, settings_cuda = get_gpu_settings()
        
        # Generate audio for debugging
        sample_rate, audio_data = generate_audio(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            use_gpu=request.use_gpu,
            breathiness=request.breathiness,
            tenseness=request.tenseness,
            jitter=request.jitter,
            sultry=request.sultry
        )
        
        # Tokenize text
        phonemes = tokenize_text(request.text)
        
        # Get system info
        system_info = {
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if cuda_available else "N/A",
            "current_device": str(torch.cuda.current_device()) if cuda_available else "N/A",
            "torch_version": torch.__version__,
            "models_loaded": len(models),
            "pipelines_loaded": len(pipelines),
            "voices_loaded": len(voices)
        }
        
        # Get memory info if CUDA is available
        if cuda_available:
            memory_info = {
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cuda_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "cuda_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            }
            safe_dict_update(system_info, memory_info)
        
        return DebugResponse(
            success=True,
            audio_length=len(audio_data),
            sample_rate=sample_rate,
            phoneme_length=len(phonemes),
            system_info=system_info,
            phonemes=phonemes
        )
    
    try:
        result = safe_execute(debug_pipeline_safe, context="TTS pipeline debug")
        log_operation_success("TTS pipeline debug", voice=request.voice)
        return result
    except Exception as e:
        log_operation_failure("TTS pipeline debug", e, voice=request.voice)
        return DebugResponse(
            success=False,
            error=str(e),
            audio_length=0,
            sample_rate=0,
            phoneme_length=0,
            system_info={},
            phonemes=""
        )

@router.get("/gpu-info")
async def get_gpu_info():
    """Get detailed GPU information"""
    log_operation_start("GPU info check")
    
    def get_gpu_info_safe():
        cuda_available, settings_cuda = get_gpu_settings()
        
        gpu_info = {
            "cuda_available": cuda_available,
            "settings_cuda_enabled": settings_cuda,
            "torch_version": torch.__version__
        }
        
        if cuda_available:
            cuda_info = {
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "memory_cached_gb": torch.cuda.memory_reserved() / 1024**3
            }
            safe_dict_update(gpu_info, cuda_info)
        
        return gpu_info
    
    try:
        result = safe_execute(get_gpu_info_safe, context="GPU info check")
        log_operation_success("GPU info check", cuda_available=result["cuda_available"])
        return result
    except Exception as e:
        log_operation_failure("GPU info check", e)
        raise handle_http_error(e, context="GPU info check")
