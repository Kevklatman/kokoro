"""
TTS model management and initialization
"""
import os
import sys
import torch
from typing import Dict

# Configure logger with fallback to print
try:
    from loguru import logger
except ImportError:
    # Create a simple logger fallback using print
    class PrintLogger:
        def info(self, msg): print(f"INFO: {msg}")
        def error(self, msg): print(f"ERROR: {msg}")
        def warning(self, msg): print(f"WARNING: {msg}")
        def debug(self, msg): print(f"DEBUG: {msg}")
    logger = PrintLogger()

from huggingface_hub import login
from kokoro import KModel, KPipeline
from entry.config import get_settings


# Global model storage
models = {}
pipelines = {}
VOICES = set()

# Voice choices and presets
CHOICES = {
    '🇺🇸 🚺 Heart ❤️': 'af_heart',
    '🇺🇸 🚺 Bella 🔥': 'af_bella',
    '🇺🇸 🚺 Nicole 🎧': 'af_nicole',
    '🇺🇸 🚺 Aoede': 'af_aoede',
    '🇺🇸 🚺 Kore': 'af_kore',
    '🇺🇸 🚺 Sarah': 'af_sarah',
    '🇺🇸 🚺 Nova': 'af_nova',
    '🇺🇸 🚺 Sky': 'af_sky',
    '🇺🇸 🚺 Alloy': 'af_alloy',
    '🇺🇸 🚺 Jessica': 'af_jessica',
    '🇺🇸 🚺 River': 'af_river',
    '🇺🇸 🚹 Michael': 'am_michael',
    '🇺🇸 🚹 Fenrir': 'am_fenrir',
    '🇬🇧 🚹 Daniel': 'bm_daniel',
}

# Official voice presets
VOICE_PRESETS = {
    'literature': {
        'voice': 'af_bella',
        'speed': 1.1,
        'breathiness': 0.1,
        'tenseness': 0.1,
        'jitter': 0.15,
        'sultry': 0.1
    },
    'articles': {
        'voice': 'af_sky',
        'speed': 1.0,
        'breathiness': 0.15,
        'tenseness': 0.5,
        'jitter': 0.3,
        'sultry': 0.1
    }
}


def initialize_models(force_online=False):
    """Initialize TTS models and pipelines"""
    global models, pipelines, VOICES
    
    settings = get_settings()
    
    # Login to Hugging Face if token is provided
    if settings.hf_token:
        login(token=settings.hf_token)
    
    print(f"Using models directory: {settings.models_dir}")
    
    # Initialize models
    cuda_available = torch.cuda.is_available() and settings.cuda_available
    
    # Save original offline mode setting
    original_offline_mode = settings.offline_mode
    
    try:
        # Temporarily disable offline mode for first-time initialization if needed
        if force_online:
            logger.info("Temporarily enabling online mode for model initialization")
            settings.offline_mode = False
            
        models[False] = KModel(models_dir=settings.models_dir).to('cpu').eval()
        if cuda_available:
            models[True] = KModel(models_dir=settings.models_dir).to('cuda').eval()
    except Exception as e:
        import traceback
        logger.error(f"Error initializing models: {str(e)}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        raise
    finally:
        # Restore original offline mode setting
        settings.offline_mode = original_offline_mode
    
    # Initialize pipelines
    for lang_code in 'ab':
        pipelines[lang_code] = KPipeline(
            lang_code=lang_code, 
            model=False, 
            models_dir=settings.models_dir
        )
    
    # Set lexicon entries
    pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
    pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    
    # Load available voices
    available_voices = set()
    initial_voices = set(CHOICES.values())
    
    for voice in initial_voices:
        try:
            pipelines[voice[0]].load_voice(voice)
            available_voices.add(voice)
            print(f"Successfully loaded voice: {voice}")
        except Exception as e:
            print(f"Failed to load voice {voice}: {str(e)}")
    
    VOICES.update(available_voices)
    print(f"Loaded {len(VOICES)} voices successfully")


def get_models():
    """Get initialized models"""
    return models


def get_pipelines():
    """Get initialized pipelines"""
    return pipelines


def get_voices():
    """Get available voices"""
    return VOICES


def get_voice_choices():
    """Get voice choices mapping"""
    return CHOICES


def get_voice_presets():
    """Get voice presets"""
    return VOICE_PRESETS