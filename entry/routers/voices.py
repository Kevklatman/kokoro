"""
Voice management API routes
"""
from fastapi import APIRouter, HTTPException

from entry.models import VoicesResponse, VoiceChoicesResponse, VoicePresetsResponse
from entry.core.models import get_voices, get_voice_choices, get_voice_presets

router = APIRouter()


@router.get("/voices", response_model=VoicesResponse)
async def list_voices():
    """List all available voices"""
    voices = get_voices()
    return VoicesResponse(voices=list(voices))


@router.get("/voice-choices", response_model=VoiceChoicesResponse)
async def list_voice_choices():
    """List user-friendly voice choices (display name and id)"""
    choices = get_voice_choices()
    return VoiceChoicesResponse(choices=choices)


@router.get("/voice-presets", response_model=VoicePresetsResponse)
async def list_voice_presets():
    """List available voice presets and their parameters"""
    presets = get_voice_presets()
    return VoicePresetsResponse(presets=presets)


@router.get("/voice-presets/{preset_name}")
async def get_voice_preset(preset_name: str):
    """Get a specific voice preset"""
    presets = get_voice_presets()
    if preset_name in presets:
        return {"preset": presets[preset_name]}
    else:
        raise HTTPException(status_code=404, detail="Preset not found")


@router.get("/choices", response_model=VoiceChoicesResponse)
async def list_choices():
    """List user-friendly voice choices (display name and id) - alias for voice-choices"""
    choices = get_voice_choices()
    return VoiceChoicesResponse(choices=choices)