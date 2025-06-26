"""
Unified model loading module for Kokoro TTS
Acts as a single source of truth for model loading logic across both
the kokoro core library and the entry API endpoints.
"""

import os
import io
import pickle
import torch
from typing import Optional, Dict, Any, Callable, List, Union, Tuple
from loguru import logger
from huggingface_hub import hf_hub_download, HfFileSystem

# Global variable to track offline mode
_OFFLINE_MODE = False

class IgnoreKeyUnpickler(pickle.Unpickler):
    """Custom unpickler that ignores specific keys during unpickling to handle legacy serialization issues.
    
    This is particularly useful for models serialized with older PyTorch versions that may contain
    keys (like 'v') that are incompatible with newer PyTorch versions.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.key_errors = 0
        
    def persistent_load(self, pid):
        # Adapt for any custom persistence loading
        return pid
    
    def load(self):
        """Override the load method to skip bad keys."""
        try:
            return super().load()
        except KeyError as e:
            if str(e).strip("'") == "v":
                # This is the common legacy PyTorch serialization issue
                self.key_errors += 1
                # Return empty dict as partial result
                logger.warning(f"Ignored 'v' key error during unpickling (count: {self.key_errors})")
                return {}
            else:
                # Re-raise other key errors
                logger.error(f"Failed to load due to unknown key error: {str(e)}")
                raise


def set_offline_mode(offline: bool = True):
    """Set the global offline mode for model loading.
    
    Args:
        offline: If True, models will only be loaded from local files
    """
    global _OFFLINE_MODE
    prev_mode = _OFFLINE_MODE
    _OFFLINE_MODE = offline
    return prev_mode


def get_offline_mode() -> bool:
    """Get the current offline mode setting.
    
    Returns:
        Current offline mode setting
    """
    return _OFFLINE_MODE


# Store the original torch.load function to prevent recursion
_original_torch_load = torch.load


def _load_with_direct_unpickler(file_path: str, map_location: Optional[str] = None) -> Dict:
    """Load PyTorch model directly with custom unpickler, bypassing torch.load
    
    Args:
        file_path: Path to the model file
        map_location: Device to load model to (cpu, cuda)
        
    Returns:
        Loaded model data
    """
    with open(file_path, 'rb') as f:
        unpickler = IgnoreKeyUnpickler(f)
        model_data = unpickler.load()
        # Process data for device mapping if needed
        if map_location is not None and isinstance(model_data, dict) and 'state_dict' in model_data:
            model_data['state_dict'] = {k: v.to(map_location) 
                                       for k, v in model_data['state_dict'].items()}
        return model_data


def load_model_safely(file_path: str, map_location: Optional[str] = None, **kwargs) -> Dict:
    """Safely load a PyTorch model with fallback methods to handle corrupted files and version incompatibilities.
    
    This function implements multiple loading strategies to maximize compatibility:
    1. Standard torch.load with weights_only=True
    2. Custom unpickler with torch.load
    3. Direct unpickler without torch.load
    
    Args:
        file_path: Path to the model file
        map_location: Device to load model to (cpu, cuda)
        **kwargs: Additional arguments to pass to torch.load
        
    Returns:
        Loaded PyTorch model data
    
    Raises:
        Exception: If all loading strategies fail
    """
    logger.info(f"Loading model from {file_path} with map_location={map_location}")
    
    # Multiple loading strategies in order of preference
    loading_strategies = [
        # Strategy 1: Use original loader with weights_only (works with newer PyTorch)
        lambda: _original_torch_load(file_path, map_location=map_location, weights_only=True, **kwargs),
        
        # Strategy 2: Use custom unpickler explicitly
        lambda: _original_torch_load(file_path, map_location=map_location, pickle_module=pickle, 
                        pickle_load_args={'unpickler': IgnoreKeyUnpickler}, **kwargs),
        
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


def cached_hub_download(repo_id: str, filename: str, models_dir: Optional[str] = None, 
                        force_download: bool = False, **kwargs) -> str:
    """Download a file from Hugging Face Hub or use cached version when available
    
    Args:
        repo_id: The Hugging Face repository ID
        filename: Name of the file to download
        models_dir: Directory to save models to
        force_download: If True, force re-download even if file exists
        **kwargs: Additional arguments to pass to hf_hub_download
        
    Returns:
        Path to the downloaded/cached file
        
    Raises:
        ValueError: If in offline mode and file not found locally
    """
    # Check offline mode
    if _OFFLINE_MODE:
        if models_dir:
            local_path = os.path.join(models_dir, repo_id.split('/')[-1], filename)
            if os.path.exists(local_path):
                logger.debug(f"Using local file in offline mode: {local_path}")
                return local_path
        # No models_dir or file not found
        raise ValueError(f"Cannot download {filename} from {repo_id} in offline mode")
    
    # Use models_dir if provided, else let HF use default cache
    local_dir = os.path.join(models_dir, repo_id.split('/')[-1]) if models_dir else None
    cache_dir = local_dir if local_dir else None
    
    # Check for existing file
    if local_dir and not force_download:
        local_path = os.path.join(local_dir, filename)
        if os.path.exists(local_path):
            logger.debug(f"Using cached file: {local_path}")
            return local_path
    
    # Ensure directory exists
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
    
    # Download from Hugging Face
    logger.info(f"Downloading {filename} from {repo_id}")
    try:
        file_path = hf_hub_download(
            repo_id=repo_id, 
            filename=filename,
            cache_dir=cache_dir,
            **kwargs
        )
        logger.info(f"Successfully downloaded to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to download {filename} from {repo_id}: {str(e)}")
        raise


def replace_model_load():
    """Monkey patch torch.load with our safe loading function"""
    torch.load = load_model_safely
    
    
def restore_model_load():
    """Restore original torch.load function"""
    torch.load = _original_torch_load


class SafeModelLoader:
    """Context manager to temporarily replace torch.load with safe version
    
    Usage:
        with SafeModelLoader():
            model = torch.load('model.pt')
    """
    
    def __enter__(self):
        self.original_load = torch.load
        torch.load = load_model_safely
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.load = self.original_load
