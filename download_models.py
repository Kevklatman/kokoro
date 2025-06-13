import os
import sys
from huggingface_hub import hf_hub_download, hf_hub_url, snapshot_download, HfApi, HfFolder
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Enable faster downloads
if sys.version_info < (3, 7):
    raise RuntimeError("Python 3.7 or higher is required to run this script.")  # Ensure Python version is compatible



# List all required files here
MODEL_REPO = "hexgrad/Kokoro-82M"

token = os.environ.get("HF_TOKEN")

snapshot_download(
    repo_id=MODEL_REPO,
    token=token,
    local_dir="models",
    local_dir_use_symlinks=False,  # Avoid symlinks for compatibility
)
print(MODEL_REPO)

# Download the model files
def download_model_files(repo_id, files, local_dir="models"):
    for file in files:
        url = hf_hub_url(repo_id=repo_id, filename=file)
        hf_hub_download(repo_id=repo_id, filename=file, local_dir=local_dir, token=token)
        print(f"Downloaded {file} from {url}")