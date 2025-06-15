#!/bin/bash
# Script to test model loading
from_env=${OFFLINE_MODE:-false}
echo "Testing KModel multi-stage loading approach..."
echo "OFFLINE_MODE is set to: $from_env"
python -c "from kokoro.model_utils import _set_offline_mode; from kokoro.model import KModel; import os; _set_offline_mode(False); print('Model loading test'); k = KModel(repo_id='hexgrad/Kokoro-82M', models_dir='/app/models'); print('Loading test complete')" || echo "Model loading test failed but continuing startup"
