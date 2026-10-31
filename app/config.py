"""
TRUSTLENS Deepfake Backend Configuration.
Reads from environment variables with sensible production & hackathon defaults.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Service metadata
APP_NAME = "TRUSTLENS Deepfake Detection API"
APP_VERSION = "1.0.0"

# Media upload constraints
MAX_FILE_SIZE_BYTES = int(os.getenv("MAX_FILE_SIZE_BYTES", 25 * 1024 * 1024))  # 25 MB
MAX_AUDIO_DURATION_SECONDS = float(os.getenv("MAX_AUDIO_DURATION_SECONDS", 300.0))  # 5 minutes
MIN_AUDIO_DURATION_SECONDS = float(os.getenv("MIN_AUDIO_DURATION_SECONDS", 0.5))    # 0.5 seconds

ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".wma"}
ALLOWED_AUDIO_MIMES = {
    "audio/wav", "audio/x-wav", "audio/wave",
    "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/x-m4a", "audio/m4a",
    "audio/aac", "audio/x-aac",
    "audio/flac", "audio/x-flac",
    "audio/ogg", "application/ogg"
}

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_VIDEO_MIMES = {
    "video/mp4", "video/quicktime", "video/x-msvideo", "video/x-matroska", "video/webm"
}

# Pretrained Audio Deepfake Model
# Default: Hugging Face fine-tuned Wav2Vec2 on voice cloning / ASVspoof
HF_AUDIO_MODEL_ID = os.getenv(
    "HF_AUDIO_MODEL_ID",
    "garystafford/wav2vec2-deepfake-voice-detector"
)
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", str(BASE_DIR / "model_cache"))

# Target audio format for neural model ingestion
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1  # Mono

# Classification thresholds
SYNTHETIC_RISK_THRESHOLD = float(os.getenv("SYNTHETIC_RISK_THRESHOLD", 0.65))
AUTHENTIC_RISK_THRESHOLD = float(os.getenv("AUTHENTIC_RISK_THRESHOLD", 0.35))
MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.60))

# Device selection: Defaults to CPU for normal laptops
DEVICE = os.getenv("DEVICE", "cpu")

# CORS Settings
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")
