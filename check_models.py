#!/usr/bin/env python3
"""
Enhanced script to check model files and test model loading
"""
import os
import sys
import json
import importlib.util
import traceback
from pathlib import Path

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

# Try to import torch to check version compatibility
torch_available = False
torch_version = None

try:
    import torch
    torch_available = True
    torch_version = torch.__version__
    print(f"PyTorch version: {torch_version}")
except ImportError as e:
    print(f"PyTorch import error: {e}")
except Exception as e:
    print(f"Unexpected error importing PyTorch: {e}")

# Check module files
app_dir = os.path.abspath("/app" if os.path.exists("/app") else ".")
entry_models_path = os.path.join(app_dir, "entry", "core", "models.py")
kokoro_model_path = os.path.join(app_dir, "kokoro", "model.py")
models_dir = os.environ.get("MODELS_DIR", os.path.join(app_dir, "models"))

print(f"\nChecking module files:")
print(f"✓ entry/core/models.py: {'Found' if os.path.exists(entry_models_path) else 'Missing'}")
print(f"✓ kokoro/model.py: {'Found' if os.path.exists(kokoro_model_path) else 'Missing'}")

# Check important model files
config_path = os.path.join(models_dir, "config.json")
kokoro_config_path = os.path.join(models_dir, "Kokoro-82M", "config.json")
model_path = os.path.join(models_dir, "Kokoro-82M", "kokoro-v1_0.pth")
voices_dir = os.path.join(models_dir, "voices")

print(f"\nChecking model files:")
print(f"✓ Base config: {'Available' if os.path.exists(config_path) else 'Missing'}")
print(f"✓ Kokoro config: {'Available' if os.path.exists(kokoro_config_path) else 'Missing'}")
print(f"✓ Kokoro model: {'Available' if os.path.exists(model_path) else 'Missing'}")
print(f"✓ Voice models: {'Available' if os.path.exists(voices_dir) else 'Missing'}")

# Check voice files if directory exists
if os.path.exists(voices_dir):
    voice_files = [f for f in os.listdir(voices_dir) if f.endswith(".pt")]
    print(f"\nFound {len(voice_files)} voice files:")
    for voice_file in voice_files[:5]:  # Show first 5 voices
        full_path = os.path.join(voices_dir, voice_file)
        file_size = os.path.getsize(full_path)
        print(f"✓ {os.path.join(models_dir, 'voices', voice_file)}: Found ({file_size} bytes)")
    if len(voice_files) > 5:
        print(f"... and {len(voice_files) - 5} more voice files")

# Test model loading code
model_loading_works = False
print("\nModel loading code: Not ready")

print("\nModel Check Summary:")
print(f"Base config: {'Available' if os.path.exists(config_path) else 'Missing'}")
print(f"Kokoro config: {'Available' if os.path.exists(kokoro_config_path) else 'Missing'}")
print(f"Kokoro model: {'Available' if os.path.exists(model_path) else 'Missing'}")
print(f"Voice models: {'Available' if os.path.exists(voices_dir) and len(voice_files) > 0 else 'Missing'}")
print(f"Model loading code: {'Ready' if model_loading_works else 'Not ready'}")

print("\nNOTE: Missing models will be downloaded from Hugging Face at runtime.")

# Initialize module-level variables
entry_core_models_available = False
kokoro_model_available = False
models_loaded = False
kokoro_loaded = False

# Try to actually load the model
try:
    print("Attempting to verify model loading...")
    # Try direct torch.load first
    if os.path.exists(model_path):
        print("Testing KModel multi-stage loading approach...")
        # Try different loading configurations
        loading_configs = [
            {"map_location": "cpu", "weights_only": True},
            {"map_location": "cpu"},
            {"map_location": "cpu", "pickle_module": torch.serialization.pickle}
        ]
        
        success = False
        for i, config in enumerate(loading_configs, 1):
            try:
                print(f"Model loading attempt {i} with config: {config}")
                # Use direct torch.load to test file integrity
                model_data = torch.load(model_path, **config)
                print(f"✓ Successfully loaded model with config: {config}")
                success = True
                break
            except Exception as e:
                print(f"✗ Failed with config {config}: {str(e)}")
                continue
        
        if not success:
            print("❌ All direct loading attempts failed. Model file may be corrupt or incompatible.")
        
        print("\nModel Loading Check Summary:")
        print(f"Model loading: {'Successful' if success else 'Failed'}")
        
except Exception as e:
    print(f"❌ Error testing model loading: {e}")
    traceback.print_exc()

print("\nNOTE: Model loading issues will be resolved at runtime.")


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
            # Check if we can access the model loading functions
            if hasattr(sys.modules.get("kokoro.model_loader", None), "load_model_safely"):
                kokoro_loaded = True
            else:
                print("load_model_safely not found in kokoro.model_loader")
        except Exception as e:
            print(f"Could not check for load_model_safely: {e}")
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
    
    # Check for model file in the Hugging Face cache structure
    kokoro_model = False
    hf_cache_dir = kokoro_dir / "models--hexgrad--Kokoro-82M"
    if hf_cache_dir.exists():
        snapshots_dir = hf_cache_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    model_file = snapshot / "kokoro-v1_0.pth"
                    if model_file.exists():
                        kokoro_model = True
                        print(f"✓ Found model file: {model_file}")
                        break
    
    if not kokoro_model:
        print("✗ kokoro-v1_0.pth: Not found in expected locations")
    
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
    
    # Check if we can load models using the KModel multi-stage approach
    safe_loader_check = False
    try:
        # Find the actual model file path
        model_file_path = None
        hf_cache_dir = kokoro_dir / "models--hexgrad--Kokoro-82M"
        if hf_cache_dir.exists():
            snapshots_dir = hf_cache_dir / "snapshots"
            if snapshots_dir.exists():
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir():
                        potential_path = snapshot / "kokoro-v1_0.pth"
                        if potential_path.exists():
                            model_file_path = potential_path
                            break
        
        if models_loaded and model_file_path and os.path.exists(model_file_path):
            print("Attempting to verify model loading...")
            try:
                # Try multiple approaches to load the model file
                # First try using our load_model_safely if available
                if hasattr(sys.modules.get("kokoro.model_loader", {}), "load_model_safely"):
                    from kokoro.model_loader import load_model_safely
                    _ = load_model_safely(str(model_file_path), map_location='cpu')
                    safe_loader_check = True
                    print("✓ Safe model loader works correctly")
                # Also try the multi-stage approach from KModel
                elif kokoro_model_available:
                    # Use the multi-stage loading approach from KModel
                    from kokoro.model import KModel
                    print("Testing KModel multi-stage loading approach...")
                    for attempt, config in enumerate([
                        {"map_location": 'cpu', "weights_only": True},  # Safest option first
                        {"map_location": 'cpu', "weights_only": False},  # Less safe but needed for some models
                        {"map_location": 'cpu'}  # Basic compatibility mode
                    ]):
                        try:
                            print(f"Model loading attempt {attempt+1} with config: {config}")
                            _ = torch.load(model_file_path, **config)
                            print(f"✓ Successfully loaded model with config: {config}")
                            safe_loader_check = True
                            break
                        except Exception as e:
                            print(f"Failed to load model (attempt {attempt+1}): {str(e)}")
                else:
                    print("❌ Cannot use multi-stage loading, KModel not available")
            except Exception as e:
                print(f"❌ Could not load model using any loading method: {e}")
        else:
            print("❌ Cannot test model loading because models are not available or code is not ready")
    except Exception as e:
        print(f"❌ Error testing model loading: {e}")
    
    # Summary
    print("\nModel Loading Check Summary:")
    print(f"Model loading: {'Successful' if safe_loader_check else 'Failed'}")
    
    if offline_mode:
        if not safe_loader_check:
            print("\nERROR: Model loading failed in OFFLINE_MODE")
            print("       The application will likely fail to start.")
            return 1
    else:
        print("\nNOTE: Model loading issues will be resolved at runtime.")
        if not hf_token:
            print("      But no HF_TOKEN is set, so private models won't be accessible.")
    
    # Always return 0 for non-offline mode, even if model loading test fails
    # This allows the container to start and download models at runtime
    return 0

if __name__ == "__main__":
    sys.exit(main())
