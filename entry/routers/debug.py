"""
Debug router for TTS functionality
"""
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel
from loguru import logger
from typing import Dict, Any, Optional

from entry.core.tts import preprocess_text, generate_audio, select_voice_and_preset
from entry.config import get_settings

router = APIRouter(
    prefix="/debug",
    tags=["debug"],
)

class DebugTTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    fiction: bool = False

class DebugResponse(BaseModel):
    status: str
    details: Dict[str, Any]

@router.post("/tts-pipeline")
async def debug_tts_pipeline(request: DebugTTSRequest) -> DebugResponse:
    """Debug the TTS pipeline with detailed logging"""
    try:
        # Process similar to the normal TTS route but with more logging
        text = request.text
        voice = request.voice
        fiction = request.fiction
        
        logger.info(f"Debug TTS request for voice {voice} with fiction={fiction}")
        
        # Preprocess the text
        preprocessed_text = preprocess_text(text)
        logger.info(f"Preprocessed text: '{preprocessed_text}'")
        
        # Select voice and preset
        selected_voice, emotion_preset = select_voice_and_preset(voice, fiction=fiction)
        logger.info(f"Selected voice: {selected_voice}, preset: {emotion_preset}")
        
        # Set parameters from preset
        if emotion_preset:
            speed = emotion_preset['speed']
            breathiness = emotion_preset['breathiness']
            tenseness = emotion_preset['tenseness']
            jitter = emotion_preset['jitter']
            sultry = emotion_preset['sultry']
        else:
            speed = 1.0
            breathiness = 0.0
            tenseness = 0.0
            jitter = 0.0
            sultry = 0.0
            
        # Use GPU if available based on settings
        settings = get_settings()
        use_gpu = settings.cuda_available
        
        # Try to generate audio
        try:
            logger.info("Starting audio generation...")
            result, phonemes = generate_audio(
                preprocessed_text,
                selected_voice,
                speed,
                use_gpu,
                breathiness,
                tenseness,
                jitter,
                sultry
            )
            
            if result is None:
                logger.error("Audio generation returned None")
                return DebugResponse(
                    status="error",
                    details={
                        "message": "Audio generation returned None",
                        "preprocessed_text": preprocessed_text,
                        "selected_voice": selected_voice,
                        "phonemes": phonemes
                    }
                )
            
            sample_rate, audio_data = result
            logger.info(f"Audio generated successfully! Sample rate: {sample_rate}, shape: {audio_data.shape}")
            
            return DebugResponse(
                status="success",
                details={
                    "sample_rate": sample_rate,
                    "audio_length": len(audio_data),
                    "preprocessed_text": preprocessed_text,
                    "selected_voice": selected_voice,
                    "phoneme_length": len(phonemes)
                }
            )
        
        except Exception as e:
            logger.exception("Error during audio generation")
            return DebugResponse(
                status="error", 
                details={
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "preprocessed_text": preprocessed_text,
                    "selected_voice": selected_voice
                }
            )
            
    except Exception as e:
        logger.exception("Debug endpoint failed")
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")
