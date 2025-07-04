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
    try:
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
            system_info.update({
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cuda_memory_reserved": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "cuda_memory_cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB"
            })
        
        return DebugResponse(
            success=True,
            audio_length=len(audio_data),
            sample_rate=sample_rate,
            phoneme_length=len(phonemes),
            system_info=system_info,
            phonemes=phonemes
        )
        
    except Exception as e:
        logger.error(f"Debug TTS pipeline error: {str(e)}")
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
async def get_gpu_info() -> Dict[str, Any]:
    """Get GPU and device information"""
    try:
        settings = get_settings()
        info = {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "settings_cuda_available": settings.cuda_available,
            "current_device": str(torch.cuda.current_device()) if torch.cuda.is_available() else "N/A",
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total"] = torch.cuda.get_device_properties(0).total_memory
            info["gpu_memory_allocated"] = torch.cuda.memory_allocated(0)
            info["gpu_memory_cached"] = torch.cuda.memory_reserved(0)
        
        return info
    except Exception as e:
        logger.exception("Failed to get GPU info")
        raise HTTPException(status_code=500, detail=f"GPU info failed: {str(e)}")
