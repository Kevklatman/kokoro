"""
TTS model management and initialization
"""
import os
import sys
import torch
import io
import pickle
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



# Define a custom unpickler to ignore the problematic 'v' key
class IgnoreKeyUnpickler(pickle.Unpickler):
    """Custom unpickler that ignores 'v' key errors and other legacy serialization issues.
    
    This handles legacy PyTorch serialization issues between versions.
    """
    def __init__(self, file_obj):
        super().__init__(file_obj)
        self.key_errors = 0
        self.ignored_errors = []
    
    def find_class(self, module, name):
        # Handle class lookup for legacy serialized models
        try:
            return super().find_class(module, name)
        except (ImportError, AttributeError) as e:
            error_msg = f"Ignored error during unpickling - module: {module}, name: {name}, error: {e}"
            logger.warning(error_msg)
            self.ignored_errors.append(error_msg)
            # Return a placeholder for missing classes
            return lambda *args, **kwargs: None
    
    def persistent_load(self, pid):
        # Handle legacy storage references
        try:
            if isinstance(pid, tuple) and pid[0] == 'storage':
                return pid
            logger.debug(f"Persistent load called with: {type(pid)} {str(pid)[:100]}")
            return pid
        except Exception as e:
            logger.warning(f"Handled error in persistent_load: {e}")
            return None
            
    def load_build(self, *args, **kwargs):
        try:
            return super().load_build(*args, **kwargs)
        except KeyError as e:
            if "'v'" in str(e):
                self.key_errors += 1
                logger.info(f"Ignoring 'v' key error in unpickle load_build (occurrence {self.key_errors})")
                # Return None for this particular failed build
                return None
            # For other key errors, log but still raise
            logger.error(f"Unhandled key error in unpickler: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in load_build: {e}")
            raise
            
    def load(self):
        try:
            return super().load()
        except Exception as e:
            # Last resort error handling
            logger.error(f"Error in main unpickler load: {e}")
            if self.key_errors > 0:
                logger.warning(f"Model load partially succeeded with {self.key_errors} ignored 'v' key errors")
                # Try to return a partial result if we've been handling 'v' keys
                return {}
            raise

# Store the original torch.load function to prevent recursion
_original_torch_load = torch.load

# Direct function to load models safely - completely independent of the normal torch.load
def load_model_safely(file_path, map_location=None, **kwargs):
    """Safely load a PyTorch model with fallback methods to handle corrupted files
    
    Args:
        file_path: Path to the model file
        map_location: Device to load model to (cpu, cuda)
        **kwargs: Additional arguments to pass to torch.load
        
    Returns:
        Loaded PyTorch model data
    """
    logger.info(f"Loading model from {file_path} with map_location={map_location}")
    
    # Save original pickle module
    orig_pickle = pickle
    
    # Multiple loading strategies in order of preference
    loading_strategies = [
        # Strategy 1: Use original loader with weights_only
        lambda: _original_torch_load(file_path, map_location=map_location, weights_only=True, **kwargs),
        
        # Strategy 2: Use custom unpickler explicitly
        lambda: _original_torch_load(
            file_path, 
            map_location=map_location,
            pickle_module=pickle,
            pickle_load_args={'unpickler': IgnoreKeyUnpickler},
            **kwargs
        ),
        
        # Strategy 3: Direct file handling with custom unpickler
        lambda: _load_with_direct_unpickler(file_path, map_location=map_location)
    ]
    
    # Try each strategy in order
    last_error = None
    for i, strategy in enumerate(loading_strategies):
        try:
            logger.info(f"Trying loading strategy {i+1}")
            result = strategy()
            logger.info(f"Strategy {i+1} succeeded!")
            return result
        except Exception as e:
            logger.warning(f"Strategy {i+1} failed: {str(e)}")
            last_error = e
    
    # If all strategies fail, raise the last error
    logger.error(f"All loading strategies failed for {file_path}")
    raise last_error


def _load_with_direct_unpickler(file_path, map_location=None):
    """Load a model using direct file handling and custom unpickler
    for maximum compatibility with corrupted files"""
    logger.info(f"Attempting direct file unpickler strategy for {file_path}")
    
    with open(file_path, 'rb') as f:
        unpickler = IgnoreKeyUnpickler(f)
        result = unpickler.load()
        
        # Handle legacy torch storage metadata if present
        if map_location is not None and isinstance(result, dict) and '_metadata' in result:
            for key in result:
                if isinstance(result[key], torch.storage._UntypedStorage):
                    result[key] = torch.UntypedStorage.from_buffer(
                        result[key].numpy(), byte_order='little')
            
            result = torch._utils._rebuild_tensor_v2(result, map_location)
            
        return result


def initialize_models(force_online=False):
    """Initialize TTS models and pipelines"""
    global models, pipelines, VOICES
    
    # Log PyTorch version for debugging
    logger.info(f"Using PyTorch version: {torch.__version__}")
    
    settings = get_settings()
    
    # Login to Hugging Face if token is provided
    # If no token or login fails, continue with limited functionality
    auth_success = False
    
    if settings.hf_token:
        logger.info("HF token found, attempting to login")
        try:
            login(token=settings.hf_token)
            logger.info("Successfully logged into Hugging Face")
            auth_success = True
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face: {str(e)}")
            logger.warning("Continuing with limited functionality (some models may not be available)")
    else:
        logger.warning("No HF token provided, will use public access with limited functionality")
    
    logger.info(f"Using models directory: {settings.models_dir}")
    
    # Initialize models
    cuda_available = torch.cuda.is_available() and settings.cuda_available
    
    # Save original offline mode setting
    original_offline_mode = settings.offline_mode
    
    try:
        # Temporarily disable offline mode for first-time initialization if needed
        if force_online:
            logger.info("Temporarily enabling online mode for model initialization")
            settings.offline_mode = False
        
        # Override KModel initialization method to fix the load issue
        logger.info("Initializing models with safe loader")
        
        try:
            # Create a custom version of KModel that uses our safe_loader
            from kokoro.model import KModel
            
            # Save the original __init__ method
            original_init = KModel.__init__
            
            # Define a new initialization method
            def safe_init(self, repo_id=None, config=None, model=None, disable_complex=False, models_dir=None):
                # Save the original torch.load
                original_torch_load = torch.load
                
                # Replace torch.load temporarily
                torch.load = load_model_safely
                
                try:
                    # Call the original init with all parameters
                    original_init(self, repo_id=repo_id, config=config, model=model, 
                                  disable_complex=disable_complex, models_dir=models_dir)
                finally:
                    # Restore torch.load
                    torch.load = original_torch_load
            
            # Replace the initialization method
            KModel.__init__ = safe_init
            
            # Initialize models with fallback to online mode
            logger.info("Creating CPU model with safe loader")
            try:
                # Pass repo_id and models_dir separately
                models[False] = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=settings.models_dir).to('cpu').eval()
            except RuntimeError as e:
                if "not found locally" in str(e) and not settings.offline_mode:
                    logger.warning(f"Model not found locally: {str(e)}")
                    logger.warning("Attempting to download from Hugging Face...")
                    # Force online download
                    from kokoro.model_utils import _set_offline_mode
                    _set_offline_mode(False)
                    # Pass repo_id and models_dir separately on retry
                    models[False] = KModel(repo_id='hexgrad/Kokoro-82M', models_dir=settings.models_dir).to('cpu').eval()
                else:
                    raise
                    
            if cuda_available:
                logger.info("Creating CUDA model with safe loader")
                models[True] = KModel(models_dir=settings.models_dir).to('cuda').eval()
            
            logger.info("Model initialization successful")
            
            # Restore the original init
            KModel.__init__ = original_init
        except Exception as e:
            import traceback
            logger.error(f"Model initialization failed: {str(e)}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
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
            
            # Voice loading with safe loader
            try:
                # Override the pipeline's load_voice method
                original_load_voice = pipelines[voice[0]].load_voice
                
                # Create a safer version of load_voice
                def safe_load_voice(voice_name, *args, **kwargs):
                    try:
                        # First try the original method
                        return original_load_voice(voice_name, *args, **kwargs)
                    except Exception as e:
                        if "invalid load key, 'v'" in str(e):
                            # If we get the invalid key error, use our safe loader directly
                            logger.info(f"Using safe loader for voice: {voice_name}")
                            voice_model = load_model_safely(voice_path, map_location='cpu')
                            pipelines[voice[0]].voices[voice_name] = voice_model
                        else:
                            raise
                
                # Replace the method temporarily
                pipelines[voice[0]].load_voice = safe_load_voice
                
                try:
                    # Call the patched method
                    pipelines[voice[0]].load_voice(voice)
                    available_voices.add(voice)
                    logger.info(f"Successfully loaded voice: {voice}")
                except Exception as voice_error:
                    # Final attempt - direct loading
                    logger.warning(f"Voice loading failed: {str(voice_error)}")
                    logger.warning(f"Attempting direct loading for {voice}")
                    
                    # Use our safe loader directly
                    voice_model = load_model_safely(voice_path, map_location='cpu')
                    
                    # Manual registration
                    pipelines[voice[0]].voices[voice] = voice_model
                    available_voices.add(voice)
                    logger.info(f"Successfully loaded voice with direct loading: {voice}")
                finally:
                    # Restore the original method
                    pipelines[voice[0]].load_voice = original_load_voice
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