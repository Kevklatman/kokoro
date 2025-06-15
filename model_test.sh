#!/bin/bash
# Script to test model loading with comprehensive diagnostics

# Capture environment variables with defaults
from_env=${OFFLINE_MODE:-false}
models_dir=${MODELS_DIR:-/app/models}
repo_id="hexgrad/Kokoro-82M"

echo "==== MODEL LOADING TEST START ===="
echo "Environment variables:"
echo "OFFLINE_MODE: $from_env"
echo "MODELS_DIR: $models_dir"
echo "PYTHONPATH: $PYTHONPATH"

# Check file existence first
echo "\nChecking model files existence:"
if [ -d "$models_dir" ]; then
    echo "✓ Models directory exists: $models_dir"
    find "$models_dir" -type f -name "*.pt" -o -name "*.bin" | sort
else
    echo "❌ ERROR: Models directory does not exist: $models_dir"
fi

# Run Python with detailed error reporting
echo "\nTesting model loading with each strategy:"
python -c "
# Detailed multi-strategy model loading test
from kokoro.model_utils import _set_offline_mode
from kokoro.model import KModel
from entry.core.models import load_model_safely
import torch
import os
import sys

# Force colored output even in container
os.environ['LOGURU_COLORS'] = 'True'

try:
    print('\n[1/3] Strategy 1: Standard loading with explicit offline mode')
    _set_offline_mode(True)  # Force offline mode for consistent testing
    model = KModel(repo_id='$repo_id', models_dir='$models_dir')
    print('✓ Model initialized successfully')
    print(f'Model type: {type(model).__name__}')
    
    print('\n[2/3] Strategy 2: Testing model.to() functionality')
    model = model.to('cpu').eval()
    print('✓ Model successfully moved to CPU and set to eval mode')
    
    print('\n[3/3] Strategy 3: Testing direct file loading with custom unpickler')
    model_file = os.path.join('$models_dir', 'model.pt')
    if os.path.exists(model_file):
        model_data = load_model_safely(model_file, map_location='cpu')
        print(f'✓ Direct loading successful, keys: {list(model_data.keys()) if isinstance(model_data, dict) else "<not a dict>"}')
    else:
        print(f'❌ Model file not found at {model_file}')
    
    print('\nAll tests completed successfully! 🎉')
    sys.exit(0)
except Exception as e:
    print(f'\n❌ ERROR: {type(e).__name__}: {str(e)}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" || error_code=$?

# Handle Python script exit code
if [ -n "$error_code" ]; then
    echo "\n❌ Model loading test failed with exit code: $error_code"
    echo "This error is non-fatal for container startup, but indicates model loading issues."
else
    echo "\n✓ Model loading test completed successfully."
fi

echo "==== MODEL LOADING TEST END ===="

