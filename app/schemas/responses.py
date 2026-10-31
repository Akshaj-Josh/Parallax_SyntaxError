"""
Data Schemas for TRUSTLENS Deepfake Detection API responses.
Supports both Pydantic v2 (when installed) and standard library dataclasses as a fallback.
"""

from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    # Lightweight fallback for pre-install unit testing
    class BaseModel:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self) -> Dict[str, Any]:
            res = {}
            for k, v in self.__dict__.items():
                if isinstance(v, BaseModel):
                    res[k] = v.model_dump()
                elif isinstance(v, list):
                    res[k] = [item.model_dump() if isinstance(item, BaseModel) else item for item in v]
                elif isinstance(v, Enum):
                    res[k] = v.value
                else:
                    res[k] = v
            return res
        def dict(self) -> Dict[str, Any]:
            return self.model_dump()

    def Field(*args, **kwargs):
        default = kwargs.get("default", None)
        default_factory = kwargs.get("default_factory", None)
        if default_factory is not None:
            return default_factory()
        return default if default is not Ellipsis else None


class AssessmentVerdict(str, Enum):
    LIKELY_AUTHENTIC = "likely_authentic"
    POTENTIALLY_SYNTHETIC = "potentially_synthetic"
    INCONCLUSIVE = "inconclusive"


class IndicatorSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetectionIndicator(BaseModel):
    indicator: str = Field(default="", description="Name of the detected acoustic or visual artifact")
    description: str = Field(default="", description="Explanation of what this artifact signifies in synthetic media")
    severity: IndicatorSeverity = Field(default=IndicatorSeverity.LOW, description="Severity or anomaly level")
    measured_value: Optional[str] = Field(default=None, description="Observed value from the sample")
    expected_baseline: Optional[str] = Field(default=None, description="Expected range for authentic human media")


class FileMetadata(BaseModel):
    filename: str = Field(default="")
    media_type: str = Field(default="audio/wav")
    file_size_bytes: int = Field(default=0)
    duration_seconds: float = Field(default=0.0)
    sample_rate: Optional[int] = Field(default=None)
    channels: Optional[int] = Field(default=None)
    format: str = Field(default="WAV")


class ModelInferenceDetails(BaseModel):
    primary_model: str = Field(default="Audio Deepfake Detector", description="Name and source of the model")
    backend_engine: str = Field(default="Acoustic Signal Processing", description="Inference engine")
    raw_synthetic_probability: Optional[float] = Field(default=None, description="Direct model synthetic probability")
    raw_authentic_probability: Optional[float] = Field(default=None, description="Direct model authentic probability")
    feature_metrics: Dict[str, Any] = Field(default_factory=dict, description="Acoustic or spatial feature measurements")


class VerificationRecommendation(BaseModel):
    category: str = Field(default="", description="Scam vector category")
    immediate_actions: List[str] = Field(default_factory=list, description="Immediate actions to verify identity")
    what_to_avoid: List[str] = Field(default_factory=list, description="Strict warnings of what not to do")
    out_of_band_protocol: str = Field(default="", description="Independent out-of-band verification protocol")


class MediaAnalysisResponse(BaseModel):
    media_id: str = Field(default="")
    media_type: str = Field(default="audio/wav")
    file_info: FileMetadata = Field(default_factory=FileMetadata)
    overall_assessment: AssessmentVerdict = Field(default=AssessmentVerdict.INCONCLUSIVE)
    synthetic_risk_score: float = Field(default=0.5)
    model_confidence: float = Field(default=0.5)
    explanation: str = Field(default="")
    detection_indicators: List[DetectionIndicator] = Field(default_factory=list)
    verification_guidance: VerificationRecommendation = Field(default_factory=VerificationRecommendation)
    inference_details: ModelInferenceDetails = Field(default_factory=ModelInferenceDetails)
    disclaimer: str = Field(
        default=(
            "TRUSTLENS synthetic media scores reflect statistical pattern detection and acoustic "
            "artifact analysis. Neither risk score nor model confidence constitutes a guaranteed "
            "legal proof of fakery. Always verify critical requests through out-of-band channels."
        )
    )


class HealthResponse(BaseModel):
    status: str = Field(default="ok")
    service: str = Field(default="TRUSTLENS Deepfake API")
    version: str = Field(default="1.0.0")
    audio_detector_ready: bool = Field(default=True)
    active_model_name: str = Field(default="")
    device: str = Field(default="cpu")
    video_detector_available: bool = Field(default=False)
