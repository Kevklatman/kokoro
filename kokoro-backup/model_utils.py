"""
Utilities for model loading and caching to reduce API calls
"""
import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

# Global variable to temporarily override offline mode
_override_offline_mode = None

# Try to get settings, but don't fail if not available
try:
    from entry.config import get_settings
    settings = get_settings()
except (ImportError, AttributeError):
    # Create a dummy settings object with default values
    class DummySettings:
        offline_mode = False
    settings = DummySettings()


def _set_offline_mode(value):
    """
    Temporarily override the global offline mode setting.
    This is used to force online mode when needed.
    
    Args:
        value (bool): True for offline mode, False for online mode
    """
    global _override_offline_mode
    _override_offline_mode = value
    logger.info(f"Setting offline mode override to: {value}")

def cached_hub_download(repo_id, filename, models_dir=None, offline_mode=None, **kwargs):
    """
    Download a file from Hugging Face Hub with local caching priority.
    Always checks the local models_dir first before making API calls.
    
    Args:
        repo_id: Repository ID on Hugging Face
        filename: File to download
        models_dir: Local directory to check first (None for default caching)
        **kwargs: Additional arguments to pass to hf_hub_download
    
    Returns:
        Path to the downloaded or cached file
    """
    # If we have a models_dir, check for the file there first
    if models_dir:
        # Construct expected path in models_dir
        repo_name = repo_id.split("/")[-1]
        local_path = os.path.join(models_dir, repo_name, filename)
        
        # Check if file exists locally
        if os.path.exists(local_path):
            logger.info(f"Using cached file from models_dir: {local_path}")
            return local_path
        
        # For config.json, check at the root of models_dir too
        if filename == "config.json":
            root_config = os.path.join(models_dir, filename)
            if os.path.exists(root_config):
                logger.info(f"Using root config file: {root_config}")
                return root_config
                
        # For voice files, check in a voices subdirectory too
        if filename.startswith("voices/"):
            voice_path = os.path.join(models_dir, filename)
            if os.path.exists(voice_path):
                logger.info(f"Using cached voice file: {voice_path}")
                return voice_path
    
    # Check if offline mode is explicitly set
    is_offline = settings.offline_mode if offline_mode is None else offline_mode
    if _override_offline_mode is not None:
        is_offline = _override_offline_mode
        
    if is_offline:
        logger.info(f"Using offline mode - will only use local files for {filename}")
    else:
        logger.info(f"Using online mode - will download {filename} if not found locally")
    
    # Try to find the file in the Hugging Face cache directory with local_files_only
    try:
        return hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_files_only=True,
            **kwargs
        )
    except (HfHubHTTPError, FileNotFoundError, ValueError) as e:
        logger.warning(f"Could not find {filename} in HF cache: {str(e)}")
        # Instead of raising an error, continue with a fallback approach
    
    # Provide more robust fallback paths for common files
    if models_dir:
        # Try some alternative locations
        possible_paths = [
            os.path.join(models_dir, filename),  # Direct path under models_dir
            os.path.join(models_dir, os.path.basename(filename))  # Just the filename
        ]
        
        # For voice files, try some common patterns
        if 'voice' in filename or filename.endswith('.pt'):
            voice_name = os.path.basename(filename).replace('.pt', '')
            possible_paths.extend([
                os.path.join(models_dir, 'voices', f"{voice_name}.pt"),
                os.path.join(models_dir, 'voices', voice_name, 'model.pt')
            ])
        
        # Check each possible location
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"Found file at alternative location: {path}")
                return path
    
    # If we're in offline mode, we can't download the file
    if is_offline:
        logger.error(f"Could not find {filename} in any local directory.")
        logger.error(f"Please ensure the file is placed in the correct directory.")
        raise RuntimeError(f"File {filename} not found locally. Make sure all required models are present.")
    
    # If we're in online mode, try to download the file
    try:
        logger.info(f"Downloading {filename} from Hugging Face ({repo_id})")
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit hit (429). Consider enabling offline mode and ensuring all models are downloaded beforehand.")
        raise
