"""
Voice management API routes
"""
from fastapi import APIRouter, HTTPException

from entry.models import VoicesResponse, VoiceChoicesResponse, VoicePresetsResponse
from entry.core.models import get_voices, get_voice_presets, CHOICES

router = APIRouter()


@router.get("/", response_model=VoicesResponse)
async def list_voices():
    """Get list of available voices"""
    voices = get_voices()
    return VoicesResponse(voices=list(voices))


@router.get("/choices", response_model=VoiceChoicesResponse)
async def list_voice_choices():
    """Get voice choices mapping"""
    choices = CHOICES
    return VoiceChoicesResponse(choices=choices)


@router.get("/presets", response_model=VoicePresetsResponse)
async def list_voice_presets():
    """Get available voice presets"""
    presets = get_voice_presets()
    return VoicePresetsResponse(presets=presets)


@router.get("/presets/{preset_name}")
async def get_voice_preset(preset_name: str):
    """Get specific voice preset by name"""
    presets = get_voice_presets()
    if preset_name not in presets:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")
    return presets[preset_name]


@router.get("/list", response_model=VoiceChoicesResponse)
async def list_choices():
    """Get voice choices mapping (alias for /choices)"""
    choices = CHOICES
    return VoiceChoicesResponse(choices=choices)