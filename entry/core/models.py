"""
TTS model management and initialization
"""
import os
import torch
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


def initialize_models():
    """Initialize TTS models and pipelines"""
    global models, pipelines, VOICES
    
    settings = get_settings()
    
    # Login to Hugging Face if token is provided
    if settings.hf_token:
        login(token=settings.hf_token)
    
    print(f"Using models directory: {settings.models_dir}")
    
    # Initialize models
    cuda_available = torch.cuda.is_available() and settings.cuda_available
    models[False] = KModel(models_dir=settings.models_dir).to('cpu').eval()
    if cuda_available:
        models[True] = KModel(models_dir=settings.models_dir).to('cuda').eval()
    
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