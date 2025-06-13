"""
Utilities for model loading and caching to reduce API calls
"""
import os
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from loguru import logger

def cached_hub_download(repo_id, filename, models_dir=None, **kwargs):
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
    
    # Try to download with local_files_only first to avoid API calls
    try:
        return hf_hub_download(
            repo_id=repo_id, 
            filename=filename, 
            local_files_only=True,
            **kwargs
        )
    except (HfHubHTTPError, FileNotFoundError, ValueError) as e:
        logger.debug(f"File not found locally, will try to download: {e}")
    
    # If local_files_only fails, try to download from HF
    try:
        logger.info(f"Downloading {filename} from Hugging Face ({repo_id})")
        return hf_hub_download(repo_id=repo_id, filename=filename, **kwargs)
    except HfHubHTTPError as e:
        if e.response.status_code == 429:
            logger.error(f"Rate limit hit (429). Consider using offline mode or waiting.")
        raise
