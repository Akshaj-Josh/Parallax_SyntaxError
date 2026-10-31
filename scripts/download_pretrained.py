"""
Helper script to download and cache the pretrained audio deepfake model
from Hugging Face for offline or production inference.
"""

import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import HF_AUDIO_MODEL_ID, MODEL_CACHE_DIR


def download_model(model_id: str = HF_AUDIO_MODEL_ID, cache_dir: str = MODEL_CACHE_DIR):
    print(f"===========================================================")
    print(f"TRUSTLENS Pretrained Model Downloader")
    print(f"Target Model: {model_id}")
    print(f"Cache Directory: {cache_dir}")
    print(f"===========================================================\n")

    try:
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError:
        print("[ERROR] 'transformers' and 'torch' are required to download Hugging Face models.")
        print("Install them using: pip install torch transformers")
        sys.exit(1)

    os.makedirs(cache_dir, exist_ok=True)

    print(f"[1/2] Downloading feature extractor for {model_id}...")
    extractor = AutoFeatureExtractor.from_pretrained(model_id, cache_dir=cache_dir)
    print("  ✓ Feature extractor ready.")

    print(f"[2/2] Downloading model weights for {model_id}...")
    model = AutoModelForAudioClassification.from_pretrained(model_id, cache_dir=cache_dir)
    print("  ✓ Model weights ready.")

    print("\n[SUCCESS] Pretrained audio deepfake model is cached and ready for local inference!")


if __name__ == "__main__":
    download_model()
