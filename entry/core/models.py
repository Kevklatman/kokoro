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
from entry.utils.file_utils import (
    safe_file_exists, safe_directory_exists, ensure_directory_exists,
    list_directory_contents
)
from entry.utils.string_utils import format_list_for_display, build_path
from entry.utils.dict_utils import safe_dict_clear, safe_dict_update

# Global model storage - will be populated during initialization
models = {}
pipelines = {}
VOICES = set()

# Flag to track if voices are manually added for debug
MANUAL_VOICES_ADDED = False

# Default model - populated during initialization
default_model = None

# Voice choices and presets
CHOICES = {
    'af_sky': 'af_sky',
    'af_heart': 'af_heart'
}

# Official voice presets
VOICE_PRESETS = {
    'fiction': {
        'voice': 'af_sky',
        'speed': 1.1,
        'breathiness': 0.1,
        'tenseness': 0.1,
        'jitter': 0.15,
        'sultry': 0.1
    },
    'non-fiction': {
        'voice': 'af_heart',
        'speed': 1.0,
        'breathiness': 0.15,
        'tenseness': 0.5,
        'jitter': 0.3,
        'sultry': 0.1
    }
}

def authenticate_huggingface():
    """Authenticate with Hugging Face if token is provided"""
    settings = get_settings()
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
    else:
        logger.warning("No HF token provided, proceeding with limited functionality")
        
    return auth_success


def load_main_model(auth_success, models_dir):
    """Load the main model with fallback to online mode if needed"""
    model = None
    
    try:
        logger.info(f"Loading model from {models_dir} (offline_mode={get_offline_mode()})")
        model = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=models_dir)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # If failed with offline mode, try again with online mode if auth worked
        if get_offline_mode() and auth_success:
            set_offline_mode(False)
            logger.info("Retrying model loading in online mode")
            try:
                model = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=models_dir)
                logger.info("Online model loading succeeded")
                return model
            except Exception as fallback_error:
                logger.error(f"Error during online fallback load: {str(fallback_error)}")
                raise fallback_error
        else:
            # Re-raise original error if we can't try online mode
            raise


def setup_pipelines(model, models_dir):
    """Initialize pipelines for the given model"""
    local_pipelines = {}
    
    # Initialize pipelines for each language code
    for lang_code in 'ab':
        local_pipelines[lang_code] = KPipeline(
            lang_code=lang_code, 
            model=model, 
            models_dir=models_dir
        )
    
    # Set lexicon entries
    local_pipelines['a'].g2p.lexicon.golds['kokoro'] = 'k\u02c8Ok\u0259\u0279O'
    local_pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
    
    return local_pipelines


def load_voice_safely(voice_path, pipeline):
    """Load a voice model with multiple fallback strategies"""
    try:
        # Use SafeModelLoader context manager for voice loading
        with SafeModelLoader():
            try:
                # First try direct loading with torch.load (which is patched by SafeModelLoader)
                voice_model = torch.load(voice_path, map_location='cpu')
                return voice_model
            except Exception as e:
                logger.warning(f"Standard loading failed: {str(e)}")
                # If that fails, try the explicit load_model_safely function
                logger.warning("Attempting with explicit load_model_safely")
                return load_model_safely(voice_path, map_location='cpu')
    except Exception as e:
        logger.error(f"All voice loading attempts failed: {str(e)}")
        raise


def load_voice_packs(models_dir: str) -> set:
    """Load voice packs from models directory"""
    logger.info(f"Loading voice packs from models_dir: {repr(models_dir)}")
    voice_dir = build_path(models_dir, 'voices')
    logger.info(f"Constructed voice_dir: {repr(voice_dir)}")
    
    if not safe_directory_exists(voice_dir):
        logger.error(f"Voice directory not found: {voice_dir}")
        raise RuntimeError(f"Voice directory not found: {voice_dir}")
    
    logger.info(f"Found voice directory: {voice_dir}")
    
    # Get available voice files
    voice_files = list_directory_contents(voice_dir, ['pt'])
    logger.info(f"Available voice files: {voice_files}")
    
    # Load each voice
    for voice in CHOICES:
        voice_file = f"{voice}.pt"
        
        if voice_file not in voice_files:
            logger.warning(f"Voice file not found: {voice_file}")
            continue
        
        voice_path = build_path(voice_dir, voice_file)
        logger.info(f"Loading voice from: {voice_path}")
        
        try:
            # Try standard loading first
            logger.info(f"Loading voice using pipeline: {voice}")
            pipeline = get_pipeline_for_voice(voice)
            pipeline.get_reference_audio(voice)
            logger.info(f"Successfully loaded voice: {voice}")
            VOICES.add(voice)
            
        except Exception as voice_error:
            logger.warning(f"Standard voice loading failed: {str(voice_error)}")
            logger.warning(f"Attempting direct loading for {voice}")
            
            try:
                # Try direct loading as fallback
                load_model_safely(voice_path)
                logger.info(f"Successfully loaded voice with safe loader: {voice}")
                VOICES.add(voice)
                
            except Exception as e:
                logger.error(f"Failed to load voice {voice}: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue
    
    if not VOICES:
        logger.error("No voices were loaded successfully")
        raise RuntimeError("No voices were loaded successfully")
    
    # Add critical voices manually if missing
    critical_voices = ['af_sky', 'af_heart']
    for voice in critical_voices:
        if voice not in VOICES:
            logger.warning(f"Manually added missing critical voice: {voice}")
            VOICES.add(voice)
    
    return VOICES


def ensure_critical_voices():
    """Ensure critical voices (af_sky, af_heart) are available in VOICES set"""
    global MANUAL_VOICES_ADDED
    critical_voices = ['af_sky', 'af_heart']
    
    for voice in critical_voices:
        if voice not in VOICES:
            VOICES.add(voice)
            MANUAL_VOICES_ADDED = True
            logger.warning(f"Manually added missing critical voice: {voice}")
    
    return MANUAL_VOICES_ADDED


def initialize_models(force_online=False):
    """Initialize TTS models and pipelines using the centralized model loading functionality"""
    global models, pipelines, VOICES, default_model, MANUAL_VOICES_ADDED
    
    # Log PyTorch version for debugging
    logger.info(f"Using PyTorch version: {torch.__version__}")
    
    settings = get_settings()
    
    # Use our centralized model loader's context manager for safe loading
    with SafeModelLoader():
        try:
            # Save original offline mode setting
            original_offline_mode = get_offline_mode()
            
            # If force_online is True, temporarily override offline mode
            if force_online:
                logger.info("Temporarily forcing online mode for model initialization")
                set_offline_mode(False)
            
            # Step 1: Authenticate with Hugging Face
            auth_success = authenticate_huggingface()
            
            # Step 2: Load the main model
            model = load_main_model(auth_success, settings.models_dir)
            default_model = model  # Store as default model
            
            # Clear any existing data and start fresh
            safe_dict_clear([models, pipelines, VOICES])
            
            # Add models (ensure both CPU and GPU models exist)
            models['standard'] = model
            models[False] = model  # CPU model
            
            # Initialize GPU model if CUDA is available
            if torch.cuda.is_available() and settings.cuda_available:
                try:
                    # Create GPU model by moving to CUDA
                    gpu_model = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=settings.models_dir)
                    gpu_model.model.cuda()  # Move model to GPU
                    models[True] = gpu_model  # GPU model
                    logger.info("GPU model initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize GPU model: {e}. Using CPU model as fallback.")
                    models[True] = model  # Use CPU model as fallback
            else:
                logger.info("CUDA not available or disabled. Using CPU model for GPU requests.")
                models[True] = model  # Use CPU model as fallback
            
            # Step 3: Initialize pipelines
            local_pipelines = setup_pipelines(model, settings.models_dir)
            safe_dict_update(pipelines, local_pipelines)
            logger.info(f"Initialized pipelines: {list(pipelines.keys())}")
            
            # Step 4: Load voices
            available_voices = load_voice_packs(settings.models_dir)
            safe_dict_update(VOICES, available_voices)
            logger.info(f"Loaded {len(VOICES)} voices successfully: {', '.join(sorted(VOICES))}")
            
            # Ensure critical voices are available
            ensure_critical_voices()
                
            # Validate initialization
            if len(models) < 1 or len(pipelines) < 1 or len(VOICES) < 1:
                raise RuntimeError(f"Initialization incomplete: models={len(models)}, pipelines={len(pipelines)}, voices={len(VOICES)}")
                
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        finally:
            # Restore original offline mode setting
            set_offline_mode(original_offline_mode)


def get_models():
    """Get initialized models"""
    # Ensure models dictionary is not empty
    if not models or None in models.values():
        logger.warning("Models dict is empty or contains None values - initialization issue detected")
        if default_model is not None:
            logger.warning("Recovering from missing models with default_model")
            models[False] = default_model  # Add CPU model as fallback
    return models


def get_pipelines():
    """Get initialized pipelines"""
    # Ensure pipelines dictionary is not empty
    if not pipelines or None in pipelines.values():
        logger.warning("Pipelines dict is empty or contains None values - initialization issue detected")
        # If we need to recreate pipelines, we need both the model and the models directory
        if default_model is not None and len(models) > 0:
            try:
                logger.warning("Attempting to recover pipelines")
                settings = get_settings()
                new_pipelines = setup_pipelines(default_model, settings.models_dir)
                safe_dict_update(pipelines, new_pipelines)
            except Exception as e:
                logger.error(f"Pipeline recovery failed: {str(e)}")
    return pipelines


def get_voices():
    """Get available voices"""
    # Ensure critical voices are available
    ensure_critical_voices()
    
    # If we had to manually add voices, warn about potential issues
    if MANUAL_VOICES_ADDED:
        logger.warning("Using manually added voices - voice models may not be properly loaded")
        
    return VOICES


def get_voice_presets():
    """Get voice presets"""
    return VOICE_PRESETS