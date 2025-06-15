#!/usr/bin/env python3
"""
Script to check if model files exist in the expected locations
"""
import os
import sys
import json
import importlib.util
from pathlib import Path

# Try to import torch to check version compatibility
torch_available = False
torch_version = None

try:
    import torch
    torch_available = True
    torch_version = torch.__version__
except ImportError:
    pass

# Fix module paths
def fix_import_paths():
    """Add necessary paths to sys.path to ensure modules can be imported"""
    app_dir = os.path.abspath("/app" if os.path.exists("/app") else ".")
    
    # Add potential module paths
    potential_paths = [app_dir, os.path.dirname(app_dir)]
    for path in potential_paths:
        if path not in sys.path:
            sys.path.insert(0, path)
            print(f"Added {path} to sys.path")

# Apply the path fix
fix_import_paths()

# Try to import our model loading functions
models_loaded = False
kokoro_loaded = False
entry_core_models_available = False
kokoro_model_available = False

# First just check if the files exist
app_dir = os.path.abspath("/app" if os.path.exists("/app") else ".")
entry_models_path = os.path.join(app_dir, "entry", "core", "models.py")
kokoro_model_path = os.path.join(app_dir, "kokoro", "model.py")

print(f"Checking for module files:")
print(f"  entry/core/models.py: {os.path.exists(entry_models_path)}")
print(f"  kokoro/model.py: {os.path.exists(kokoro_model_path)}")

try:
    try:
        # Try direct imports first
        import entry.core.models
        entry_core_models_available = True
    except ImportError as e:
        print(f"Direct import failed: {e}")
        
        # Try manual import
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("entry.core.models", entry_models_path)
            if spec:
                entry_core_models = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(entry_core_models)
                sys.modules["entry.core.models"] = entry_core_models
                entry_core_models_available = True
                print("Manually imported entry.core.models")
        except Exception as e:
            print(f"Manual import failed: {e}")
    
    try:
        # Try direct import for kokoro model
        from kokoro.model import KModel
        kokoro_model_available = True
    except ImportError as e:
        print(f"Direct Kokoro import failed: {e}")
        
        # Try manual import
        try:
            spec = importlib.util.spec_from_file_location("kokoro.model", kokoro_model_path)
            if spec:
                kokoro_model = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(kokoro_model)
                sys.modules["kokoro.model"] = kokoro_model
                kokoro_model_available = True
                print("Manually imported kokoro.model")
        except Exception as e:
            print(f"Manual Kokoro import failed: {e}")
    
    # Check if both modules are available
    if entry_core_models_available and kokoro_model_available:
        models_loaded = True
        try:
            # Check if we can access the model initialization function
            if hasattr(sys.modules.get("entry.core.models", None), "safe_load_model"):
                kokoro_loaded = True
            else:
                print("safe_load_model not found in entry.core.models")
        except Exception as e:
            print(f"Could not check for safe_load_model: {e}")
            kokoro_loaded = False
except Exception as e:
    print(f"Could not import model modules: {e}")

def check_file_exists(path, required=True):
    path = Path(path)
    exists = path.exists()
    file_size = path.stat().st_size if exists else 0
    status = "✓" if exists else "✗"
    
    print(f"{status} {path}: {'Found' if exists else 'Not found'} ({file_size} bytes)")
    
    if required and not exists:
        return False
    return True

def get_file_content(path):
    try:
        with open(path, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def main():
    # Check for environment variables
    hf_token = os.environ.get("HF_TOKEN", None)
    offline_mode = os.environ.get("OFFLINE_MODE", "false").lower() == "true"
    
    print("\n===== Kokoro TTS Model Check =====\n")
    
    print("Environment Configuration:")
    print(f"OFFLINE_MODE: {offline_mode}")
    print(f"HF_TOKEN: {'Set' if hf_token else 'Not set'}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"Working directory: {os.getcwd()}")
    
    print("\nPython Package Status:")
    print(f"PyTorch: {'Installed' if torch_available else 'Not installed'} (version: {torch_version if torch_available else 'N/A'})")
    print(f"Entry Core Models: {'Available' if entry_core_models_available else 'Not available'}")
    print(f"Kokoro Model: {'Available' if kokoro_model_available else 'Not available'}")
    
    # Add our safe model loading version info
    print(f"Safe Model Loading: {'Implemented' if models_loaded else 'Not implemented'}")
    print(f"Kokoro Model Loading: {'Ready' if kokoro_loaded else 'Not ready'}")
    
    if offline_mode and not hf_token:
        print("\nWARNING: Offline mode is enabled but no HF_TOKEN is set.")
        print("         If models are missing, the application will fail.")
    
    # Check for essential model files
    app_dir = Path("/app" if os.path.exists("/app") else ".")
    models_dir = app_dir / "models"
    kokoro_dir = models_dir / "Kokoro-82M"
    
    print("\nChecking base model configuration:")
    config_found = check_file_exists(models_dir / "config.json")
    
    # If config exists, show some content
    if config_found:
        try:
            with open(models_dir / "config.json", 'r') as f:
                config = json.load(f)
                print(f"  Config type: {config.get('_name_or_path', 'unknown')}")
                print(f"  Model dimensions: {config.get('d_model', 'unknown')}")
        except Exception as e:
            print(f"  Error reading config: {e}")
    
    # Check for model-specific files
    print("\nChecking Kokoro model files:")
    kokoro_config = check_file_exists(kokoro_dir / "config.json")
    kokoro_model = check_file_exists(kokoro_dir / "kokoro-v1_0.pth")
    
    # Check model file size and integrity
    if kokoro_model:
        try:
            model_stat = (kokoro_dir / "kokoro-v1_0.pth").stat()
            print(f"  Model file size: {model_stat.st_size / (1024*1024):.2f} MB")
            # Check if file looks valid (at least not too small)
            if model_stat.st_size < 1000000:  # 1MB
                print("  WARNING: Model file seems too small, might be incomplete")
        except Exception as e:
            print(f"  Error accessing model file: {e}")
    
    # Check for voice models
    voices_dir = models_dir / "voices"
    print("\nChecking voice models:")
    if not voices_dir.exists():
        print(f"✗ {voices_dir}: Voice directory not found")
        voices_found = False
    else:
        voice_files = list(voices_dir.glob("*.pt"))
        if voice_files:
            for voice_file in voice_files:
                check_file_exists(voice_file, required=False)
            voices_found = True
        else:
            print(f"✗ {voices_dir}: No voice files found")
            voices_found = False
    
    # Summary
    print("\nModel Check Summary:")
    print(f"Base config: {'Available' if config_found else 'Missing'}")
    print(f"Kokoro config: {'Available' if kokoro_config else 'Missing'}")
    print(f"Kokoro model: {'Available' if kokoro_model else 'Missing'}")
    print(f"Voice models: {'Available' if voices_found else 'Missing'}")
    print(f"Model loading code: {'Ready' if models_loaded and kokoro_loaded else 'Not ready'}")
    
    if offline_mode:
        if not all([config_found, kokoro_config, kokoro_model]):
            print("\nERROR: Required model files missing in OFFLINE_MODE")
            print("       The application will likely fail to start.")
            return 1
    else:
        print("\nNOTE: Missing models will be downloaded from Hugging Face at runtime.")
        if not hf_token:
            print("      But no HF_TOKEN is set, so private models won't be accessible.")
    
    # Test load model if everything is available
    if models_loaded and kokoro_loaded and kokoro_model:
        print("\nTrying to verify model loading...")
        try:
            from entry.core.models import safe_load_model
            
            # Just test the loading function without initializing full model
            model_path = str(kokoro_dir / "kokoro-v1_0.pth")
            print(f"Testing safe_load_model on {model_path}")
            
            try:
                # Just attempt to load the state dict to verify
                state_dict = safe_load_model(model_path)
                print("✓ Model loading successful!")
            except Exception as e:
                print(f"✗ Model loading failed: {e}")
                return 1
        except Exception as e:
            print(f"✗ Could not test model loading: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
