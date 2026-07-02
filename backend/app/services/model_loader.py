"""
Model weight loader — downloads best.pt from HuggingFace Hub at startup
if it doesn't already exist locally.

Set these environment variables in Render dashboard:
  HF_MODEL_REPO  = your-hf-username/pricetag-ocr-model
  HF_MODEL_FILE  = best.pt  (default)
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
WEIGHTS_PATH = ROOT / "models" / "checkpoints" / "best.pt"


def ensure_weights() -> Path:
    """
    Returns path to weights file. Downloads from HuggingFace Hub if missing.
    Falls back to MockDetector path if HF_MODEL_REPO is not set.
    """
    if WEIGHTS_PATH.exists():
        return WEIGHTS_PATH

    repo = os.environ.get("HF_MODEL_REPO", "").strip()
    filename = os.environ.get("HF_MODEL_FILE", "best.pt").strip()

    if not repo:
        print("⚠️  No HF_MODEL_REPO set and no local weights found.")
        print("    Running in demo mode with MockDetector.")
        return WEIGHTS_PATH  # doesn't exist — detector.py handles this gracefully

    print(f"📥  Downloading {filename} from HuggingFace Hub: {repo} …")
    try:
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(WEIGHTS_PATH.parent),
        )
        print(f"✅  Weights saved to {downloaded}")
        return Path(downloaded)
    except Exception as e:
        print(f"❌  Failed to download weights: {e}")
        print("    Running in demo mode with MockDetector.")
        return WEIGHTS_PATH
