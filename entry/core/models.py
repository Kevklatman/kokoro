"""
TTS model management and initialization
"""
import os
import sys
import torch
import traceback
from typing import Dict, List, Any, Optional, Set
from loguru import logger

from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError

# Import centralized model loading functionality
from kokoro.model_loader import (
    load_model_safely,
    cached_hub_download,
    set_offline_mode,
    get_offline_mode,
    SafeModelLoader
)

from entry.config import get_settings
from kokoro.model import KModel
from kokoro.pipeline import KPipeline

# Global model storage
models = {}
pipelines = {}
VOICES = set()

# Voice choices and presets
CHOICES = {
    'af_sky': 'af_sky',
    'af_heart': 'af_heart'
}

# Official voice presets
VOICE_PRESETS = {
    'literature': {
        'voice': 'af_sky',
        'speed': 1.1,
        'breathiness': 0.1,
        'tenseness': 0.1,
        'jitter': 0.15,
        'sultry': 0.1
    },
    'articles': {
        'voice': 'af_heart',
        'speed': 1.0,
        'breathiness': 0.15,
        'tenseness': 0.5,
        'jitter': 0.3,
        'sultry': 0.1
    }
}



# IgnoreKeyUnpickler moved to centralized kokoro.model_loader module
            
def initialize_models(force_online=False):
    """Initialize TTS models and pipelines using the centralized model loading functionality"""
    global models, pipelines, VOICES
    
    # Log PyTorch version for debugging
    import torch
    logger.info(f"Using PyTorch version: {torch.__version__}")
    
    settings = get_settings()
    
    # Login to Hugging Face if token is provided
    # If no token or login fails, continue with limited functionality
    auth_success = False
    
    if settings.hf_token:
        logger.info("HF token found, attempting to login")
        try:
            login(token=settings.hf_token)
            auth_success = True
            logger.info("Successfully logged in to Hugging Face")
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face: {str(e)}")
            auth_success = False
    
    # Use our centralized model loader's context manager for safe loading
    with SafeModelLoader():
        try:
            # Save original offline mode setting
            original_offline_mode = get_offline_mode()
            
            # If force_online is True, temporarily override offline mode
            if force_online:
                logger.info("Temporarily forcing online mode for model initialization")
                set_offline_mode(False)
                
            # Set up model instance outside try block so it can be accessed by finally
            model = None

            # Try to load model
            try:
                logger.info(f"Loading model from {settings.models_dir} (offline_mode={get_offline_mode()})")
                
                # First try to load from local or HuggingFace
                model = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=settings.models_dir)
                logger.info("Model loaded successfully")
                models['standard'] = model
                
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}")
                # If failed with offline mode, try again with online mode if auth worked
                if get_offline_mode() and auth_success:
                    set_offline_mode(False)
                    logger.info("Retrying model loading in online mode")
                    try:
                        model = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=settings.models_dir)
                        logger.info("Online model loading succeeded")
                        models['standard'] = model
                    except Exception as fallback_error:
                        logger.error(f"Error during online fallback load: {str(fallback_error)}")
                        raise fallback_error
                else:
                    # Re-raise original error if we can't try online mode
                    raise
                    
            # Load voice files
            logger.info("Loading voice files")
            VOICES = get_voices(settings.models_dir)
                
        except Exception as e:
            import traceback
            logger.error(f"Error initializing models: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Restore original offline mode setting
            set_offline_mode(original_offline_mode)
    
    # Initialize pipelines
    for lang_code in 'ab':
        pipelines[lang_code] = KPipeline(
            lang_code=lang_code, 
            model=models['standard'], 
            models_dir=settings.models_dir
        )
    
    # Set lexicon entries
    pipelines['a'].g2p.lexicon.golds['kokoro'] = 'k\u02c8Ok\u0259\u0279O'
    pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    
    # Load available voices
    available_voices = set()
    initial_voices = set(CHOICES.values())
    
    # Check if voice files exist
    voice_dir = os.path.join(settings.models_dir, 'voices')
    if not os.path.exists(voice_dir):
        logger.error(f"Voice directory not found: {voice_dir}")
        raise RuntimeError(f"Voice directory not found: {voice_dir}")
    
    logger.info(f"Found voice directory: {voice_dir}")
    voice_files = os.listdir(voice_dir)
    logger.info(f"Available voice files: {voice_files}")
    
    for voice in initial_voices:
        try:
            voice_file = f"{voice}.pt"
            if voice_file not in voice_files:
                logger.warning(f"Voice file not found: {voice_file}")
                continue
                
            voice_path = os.path.join(voice_dir, voice_file)
            logger.info(f"Loading voice from: {voice_path}")
            
            # Voice loading with centralized safe loader
            try:
                # Use SafeModelLoader context manager for voice loading
                with SafeModelLoader():
                    try:
                        # Try standard pipeline loading first
                        logger.info(f"Loading voice using pipeline: {voice}")
                        pipelines[voice[0]].load_voice(voice)
                        available_voices.add(voice)
                        logger.info(f"Successfully loaded voice: {voice}")
                    except Exception as voice_error:
                        logger.warning(f"Standard voice loading failed: {str(voice_error)}")
                        logger.warning(f"Attempting direct loading for {voice} using centralized loader")
                        
                        # Use our centralized safe loader
                        try:
                            # Load voice model using centralized loader
                            voice_model = torch.load(voice_path, map_location='cpu')
                            
                            # Manual registration
                            pipelines[voice[0]].voices[voice] = voice_model
                            available_voices.add(voice)
                            logger.info(f"Successfully loaded voice with centralized loader: {voice}")
                        except Exception as direct_error:
                            # If that fails too, try the explicit load_model_safely function
                            logger.warning(f"Direct loading failed: {str(direct_error)}")
                            logger.warning(f"Final attempt with explicit load_model_safely for {voice}")
                            
                            voice_model = load_model_safely(voice_path, map_location='cpu')
                            pipelines[voice[0]].voices[voice] = voice_model
                            available_voices.add(voice)
                            logger.info(f"Successfully loaded voice with fallback loader: {voice}")

            except Exception as e:
                logger.error(f"All attempts to load voice {voice} failed: {str(e)}")
                import traceback
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                raise
        except Exception as e:
            logger.error(f"Failed to load voice {voice}: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    if not available_voices:
        logger.error("No voices were loaded successfully")
        raise RuntimeError("No voices were loaded successfully")
    
    VOICES.update(available_voices)
    logger.info(f"Loaded {len(VOICES)} voices successfully: {', '.join(sorted(VOICES))}")


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