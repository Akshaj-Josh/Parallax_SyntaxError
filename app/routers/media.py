"""
FastAPI Router for Media Analysis (Audio and Video Deepfake Detection).
Prefix: /api/media
"""

import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from app.config import (
    APP_NAME,
    APP_VERSION,
    DEVICE,
    AUDIO_SAMPLE_RATE
)
from app.schemas.responses import (
    MediaAnalysisResponse,
    HealthResponse,
    FileMetadata
)
from app.utils.file_security import validate_file_metadata, managed_temp_file
from app.utils.audio_utils import get_audio_info, load_and_preprocess_audio
from app.services.audio_detector import AudioDeepfakeDetector
from app.services.video_detector import VideoDeepfakeDetector
from app.services.verification_advisor import VerificationAdvisor

router = APIRouter(prefix="/api/media", tags=["Media Deepfake Detection"])

# Shared service instances
audio_detector = AudioDeepfakeDetector()
video_detector = VideoDeepfakeDetector()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns service health, active detection engines, and model readiness status.
    """
    return HealthResponse(
        status="ok",
        service=APP_NAME,
        version=APP_VERSION,
        audio_detector_ready=True,
        active_model_name=audio_detector.model_id if audio_detector.neural_model_loaded else "Physical Acoustic Signal Analyzer",
        device=DEVICE,
        video_detector_available=video_detector.is_available()
    )


@router.post("/analyze-audio", response_model=MediaAnalysisResponse)
async def analyze_audio(file: UploadFile = File(..., description="Audio recording to inspect (WAV, MP3, M4A)")):
    """
    Upload and analyze an audio file for synthetic speech, AI voice cloning, and vocoder artifacts.
    Guarantees privacy: temporary files are purged from disk immediately upon completion.
    """
    # 1. Validate file format and MIME headers
    ext = validate_file_metadata(file, media_category="audio")
    media_id = f"aud_{uuid.uuid4().hex[:12]}"

    # 2. Ephemeral file stream with size validation and automatic deletion
    with managed_temp_file(file, suffix=ext) as temp_path:
        # Extract file metadata
        meta = get_audio_info(temp_path)

        # Decode, normalize, resample to 16kHz mono float32
        waveform, sr, duration = load_and_preprocess_audio(
            temp_path,
            target_sr=AUDIO_SAMPLE_RATE
        )

        # Execute deepfake detection inference
        (
            verdict,
            risk_score,
            confidence,
            explanation,
            indicators,
            inference_details
        ) = audio_detector.analyze_audio(waveform, duration)

        # Generate contextual verification guidance
        guidance = VerificationAdvisor.get_guidance(
            verdict=verdict,
            risk_score=risk_score,
            media_type="audio"
        )

        file_info = FileMetadata(
            filename=file.filename or f"unnamed{ext}",
            media_type=file.content_type or f"audio/{ext.replace('.', '')}",
            file_size_bytes=meta["file_size_bytes"],
            duration_seconds=duration,
            sample_rate=sr,
            channels=1,
            format=meta["format"]
        )

        return MediaAnalysisResponse(
            media_id=media_id,
            media_type=file_info.media_type,
            file_info=file_info,
            overall_assessment=verdict,
            synthetic_risk_score=risk_score,
            model_confidence=confidence,
            explanation=explanation,
            detection_indicators=indicators,
            verification_guidance=guidance,
            inference_details=inference_details
        )


@router.post("/analyze-video", response_model=MediaAnalysisResponse)
async def analyze_video(file: UploadFile = File(..., description="Video file to inspect (MP4, MOV, AVI)")):
    """
    Upload and analyze a video file for facial manipulation, warping, and temporal inconsistencies.
    """
    ext = validate_file_metadata(file, media_category="video")
    media_id = f"vid_{uuid.uuid4().hex[:12]}"

    with managed_temp_file(file, suffix=ext) as temp_path:
        file_size = Path(temp_path).stat().st_size

        (
            verdict,
            risk_score,
            confidence,
            explanation,
            indicators,
            inference_details
        ) = video_detector.analyze_video(temp_path)

        guidance = VerificationAdvisor.get_guidance(
            verdict=verdict,
            risk_score=risk_score,
            media_type="video"
        )

        duration = inference_details.feature_metrics.get("video_duration_seconds", 0.0)

        file_info = FileMetadata(
            filename=file.filename or f"unnamed{ext}",
            media_type=file.content_type or f"video/{ext.replace('.', '')}",
            file_size_bytes=file_size,
            duration_seconds=duration,
            sample_rate=None,
            channels=None,
            format=ext.replace(".", "").upper()
        )

        return MediaAnalysisResponse(
            media_id=media_id,
            media_type=file_info.media_type,
            file_info=file_info,
            overall_assessment=verdict,
            synthetic_risk_score=risk_score,
            model_confidence=confidence,
            explanation=explanation,
            detection_indicators=indicators,
            verification_guidance=guidance,
            inference_details=inference_details
        )
