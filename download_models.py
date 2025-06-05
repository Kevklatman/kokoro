import os
from huggingface_hub import hf_hub_download

# List all required files here
MODEL_REPO = "hexgrad/Kokoro-82M"
FILES = [
    "config.json",
    # Add all other required files (weights, vocoder, etc)
    # e.g. "pytorch_model.bin", "vocoder.pt", etc.
]

token = os.environ.get("HF_TOKEN")

for filename in FILES:
    print(f"Downloading {filename} from {MODEL_REPO}...")
    hf_hub_download(repo_id=MODEL_REPO, filename=filename, token=token)
print("All model files downloaded.")
