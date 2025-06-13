#!/usr/bin/env python3
"""
Model Downloader Script

Downloads all required models from Hugging Face to a local directory.
This script should be run during Docker build to ensure all models are available offline.
"""
import os
import sys
import shutil
from pathlib import Path
import torch
from loguru import logger
from huggingface_hub import hf_hub_download, login

# Add project root to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from kokoro.model import KModel
from entry.config import get_settings
from kokoro.pipeline import LANG_CODES, ALIASES

def ensure_dir(directory):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)
    return directory

def download_model_files(models_dir):
    """Download all required model files"""
    # Create required directories
    ensure_dir(models_dir)
    
    # Get repository ID (using the same default as in KModel.__init__)
    repo_id = 'hexgrad/Kokoro-82M'
    
    # Download model config
    logger.info(f"Downloading model config from {repo_id}")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    # Download model weights
    logger.info(f"Downloading model weights from {repo_id}")
    model_filename = KModel.MODEL_NAMES[repo_id]
    model_path = hf_hub_download(repo_id=repo_id, filename=model_filename)
    
    # Return paths to downloaded files
    return {
        "config": config_path,
        "model": model_path
    }

def download_voices(models_dir, voices_to_download=None):
    """Download all voice files"""
    repo_id = 'hexgrad/Kokoro-82M'
    
    # If no specific voices requested, use predefined common voices
    if voices_to_download is None:
        # Define common voices for each language code
        common_voices = {
            'a': ['en_us_amy', 'en_us_andy'],     # American English
            'b': ['en_uk_libby', 'en_uk_ryan'],   # British English
            'j': ['ja_jp_yuuka', 'ja_jp_takuya'], # Japanese
            'e': ['es_es_lucia', 'es_es_pedro'],  # Spanish
        }
        
        voices_to_download = set()
        for lang_code in LANG_CODES.keys():
            if lang_code in common_voices:
                voices_to_download.update(common_voices[lang_code])
    
    # Download each voice
    voice_paths = {}
    for voice in voices_to_download:
        logger.info(f"Downloading voice: {voice}")
        try:
            voice_path = hf_hub_download(repo_id=repo_id, filename=f"voices/{voice}.pt")
            voice_paths[voice] = voice_path
        except Exception as e:
            logger.error(f"Failed to download voice {voice}: {e}")
    
    return voice_paths

def copy_to_models_dir(model_paths, models_dir):
    """Copy downloaded files to models directory with proper structure"""
    model_dir = os.path.join(models_dir, 'model')
    ensure_dir(model_dir)
    
    # Use the repo ID we've specified in the script
    repo_name = 'Kokoro-82M'  # Just the name part, not the full path
    repo_dir = os.path.join(models_dir, repo_name)
    ensure_dir(repo_dir)
    
    voices_dir = os.path.join(models_dir, 'voices')
    ensure_dir(voices_dir)
    
    # Track files we've copied
    copied_files = []
    
    for file_type, src_path in model_paths.items():
        if file_type == "config":
            # Copy to repo subdir and root of models_dir
            dst_path = os.path.join(repo_dir, "config.json")
            if not os.path.exists(dst_path) or os.path.getsize(dst_path) != os.path.getsize(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {src_path} to {dst_path}")
                copied_files.append(dst_path)
            else:
                logger.info(f"Skipping copy, file already exists: {dst_path}")
            
            dst_path = os.path.join(models_dir, "config.json")
            if not os.path.exists(dst_path) or os.path.getsize(dst_path) != os.path.getsize(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {src_path} to {dst_path}")
                copied_files.append(dst_path)
            else:
                logger.info(f"Skipping copy, file already exists: {dst_path}")
            
        elif file_type == "model":
            # Copy to repo subdir
            model_file = os.path.basename(src_path)
            dst_path = os.path.join(repo_dir, model_file)
            if not os.path.exists(dst_path) or os.path.getsize(dst_path) != os.path.getsize(src_path):
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {src_path} to {dst_path}")
                copied_files.append(dst_path)
            else:
                logger.info(f"Skipping copy, file already exists: {dst_path}")
    
    logger.info(f"Copied {len(copied_files)} new files to models directory")
    return repo_dir, voices_dir

def copy_voices(voice_paths, voices_dir):
    """Copy voice files to voices directory"""
    for voice, src_path in voice_paths.items():
        dst_path = os.path.join(voices_dir, f"{voice}.pt")
        os.system(f"cp {src_path} {dst_path}")

def main():
    """Main entry point"""
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Get settings
    settings = get_settings()
    
    # Get models directory 
    models_dir = os.environ.get("MODELS_DIR", settings.models_dir)
    logger.info(f"Using models directory: {models_dir}")
    
    # Get Hugging Face token
    hf_token = os.environ.get("HF_TOKEN", settings.hf_token)
    if not hf_token:
        logger.warning("No Hugging Face token provided. Anonymous downloads may be rate limited.")
    else:
        logger.info("Logging in to Hugging Face with token")
        login(token=hf_token)
    
    # Download model files
    model_paths = download_model_files(models_dir)
    
    # Copy model files to models directory
    repo_dir, voices_dir = copy_to_models_dir(model_paths, models_dir)
    
    # Download and copy voices
    voice_paths = download_voices(models_dir)
    copy_voices(voice_paths, voices_dir)
    
    logger.info(f"All models downloaded to {models_dir}")
    logger.info("You can now run your application in offline mode")

if __name__ == "__main__":
    main()
