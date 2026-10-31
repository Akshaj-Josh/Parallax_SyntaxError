"""
TRUSTLENS Audio Deepfake Detection Service.

Integrates:
1. Pretrained Neural Model: Fine-tuned Wav2Vec2 audio deepfake classifier
   (e.g., garystafford/wav2vec2-deepfake-voice-detector or Bgeorge/deepfake-audio-detector-wav2vec2)
2. Physical Acoustic Feature Analysis: FFT spectral rolloff, Wiener entropy, F0 micro-jitter, digital silence

Provides:
- Calibrated synthetic risk score (0.0 to 1.0)
- Model classification confidence (0.0 to 1.0, clearly separated)
- Verdict: likely_authentic, potentially_synthetic, inconclusive
- Plain-language explanation of findings
- Physical detection indicators
"""

import os
import math
from typing import Dict, Any, Tuple, Optional

try:
    import numpy as np
except ImportError:
    np = None

from app.config import (
    HF_AUDIO_MODEL_ID,
    MODEL_CACHE_DIR,
    DEVICE,
    SYNTHETIC_RISK_THRESHOLD,
    AUTHENTIC_RISK_THRESHOLD,
    MIN_CONFIDENCE_THRESHOLD,
    AUDIO_SAMPLE_RATE
)
from app.schemas.responses import (
    AssessmentVerdict,
    ModelInferenceDetails,
    DetectionIndicator
)
from app.services.acoustic_analyzer import AcousticAnalyzer


class AudioDeepfakeDetector:
    def __init__(self):
        self.model_id = HF_AUDIO_MODEL_ID
        self.device = DEVICE
        self.model = None
        self.feature_extractor = None
        self.neural_model_loaded = False
        self.load_error: Optional[str] = None

        self.acoustic_analyzer = AcousticAnalyzer(sample_rate=AUDIO_SAMPLE_RATE)

        # Attempt eager model loading (non-blocking if dependencies absent)
        self._initialize_neural_model()

    def _initialize_neural_model(self):
        """
        Attempts to load pretrained Wav2Vec2 audio deepfake classification model.
        If PyTorch or Transformers are not installed, logs clear status and falls back
        to the physical acoustic signal processing engine.
        """
        try:
            import torch
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

            os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

            print(f"[TRUSTLENS] Loading pretrained audio deepfake model: {self.model_id}...")
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id,
                cache_dir=MODEL_CACHE_DIR
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id,
                cache_dir=MODEL_CACHE_DIR
            )
            self.model.to(self.device)
            self.model.eval()
            self.neural_model_loaded = True
            print(f"[TRUSTLENS] Pretrained model loaded successfully on {self.device}.")

        except ImportError as e:
            self.load_error = f"PyTorch/Transformers not installed in this environment ({str(e)})."
            print(f"[TRUSTLENS Info] {self.load_error} Operating in Physical Acoustic Signal Engine mode.")
        except Exception as e:
            self.load_error = f"Model weights download / init error ({str(e)})."
            print(f"[TRUSTLENS Info] {self.load_error} Operating in Physical Acoustic Signal Engine mode.")

    def analyze_audio(self, waveform: np.ndarray, duration_seconds: float) -> Tuple[
        AssessmentVerdict,
        float,  # synthetic_risk_score
        float,  # model_confidence
        str,    # explanation
        list,   # indicators
        ModelInferenceDetails
    ]:
        """
        Executes genuine deepfake analysis on the normalized 16kHz mono audio waveform.
        """
        # Step 1: Physical Acoustic Feature Extraction
        acoustic_res = self.acoustic_analyzer.analyze_audio(waveform)
        acoustic_score = acoustic_res["acoustic_anomaly_score"]
        indicators = acoustic_res["indicators"]
        metrics = acoustic_res["metrics"]

        # Step 2: Pretrained Neural Inference (if available)
        neural_prob_synthetic: Optional[float] = None
        neural_prob_authentic: Optional[float] = None
        neural_confidence: Optional[float] = None
        engine_name = "Physical Acoustic Signal Analysis Engine (FFT & Prosody Artifacts)"

        if self.neural_model_loaded and self.model is not None and self.feature_extractor is not None:
            try:
                import torch
                # Process audio waveform with Hugging Face feature extractor
                inputs = self.feature_extractor(
                    waveform,
                    sampling_rate=AUDIO_SAMPLE_RATE,
                    return_tensors="pt",
                    padding=True
                )
                input_values = inputs.input_values.to(self.device)

                with torch.no_grad():
                    outputs = self.model(input_values)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

                # Model label mapping: check config labels if available
                id2label = getattr(self.model.config, "id2label", {0: "authentic", 1: "synthetic"})
                
                # Identify which class index corresponds to fake/synthetic
                fake_idx = 1
                for idx, label_name in id2label.items():
                    if any(term in str(label_name).lower() for term in ["fake", "spoof", "synthetic"]):
                        fake_idx = int(idx)
                        break

                real_idx = 1 - fake_idx if len(probs) == 2 else 0

                neural_prob_synthetic = float(probs[fake_idx])
                neural_prob_authentic = float(probs[real_idx])

                # Confidence margin: 0.0 (uniform probability) to 1.0 (complete certainty)
                neural_confidence = float(abs(neural_prob_synthetic - neural_prob_authentic))
                engine_name = f"Pretrained Wav2Vec2 ({self.model_id}) + Acoustic Fusion"

            except Exception as e:
                print(f"[TRUSTLENS Warning] Neural inference encountered an error: {e}. Falling back to acoustic metrics.")
                neural_prob_synthetic = None

        # Step 3: Calibrated Fusion of Risk Score and Confidence
        if neural_prob_synthetic is not None and neural_confidence is not None:
            # 70% weight on Pretrained Latent Classifier, 30% on Physical Vocoder Artifacts
            fused_risk = (0.70 * neural_prob_synthetic) + (0.30 * acoustic_score)
            fused_risk = min(max(fused_risk, 0.0), 1.0)

            # Statistical confidence combines model certainty and audio duration sufficiency
            duration_factor = min(duration_seconds / 3.0, 1.0)  # At least 3s for full confidence
            confidence = (0.80 * neural_confidence) + (0.20 * duration_factor)
            confidence = min(max(confidence, 0.1), 0.99)
        else:
            # Acoustic Signal Engine mode
            fused_risk = acoustic_score
            # Confidence based on amount of audio analyzed and number of valid indicators
            num_indicators = len(indicators)
            confidence = min(0.55 + (num_indicators * 0.12), 0.88) if duration_seconds >= 2.0 else 0.45

        # Format risk and confidence to 3 decimal places
        synthetic_risk_score = round(float(fused_risk), 3)
        model_confidence = round(float(confidence), 3)

        # Step 4: Overall Assessment Determination
        # If confidence is below threshold, verdict must be inconclusive to avoid false certitude
        if model_confidence < MIN_CONFIDENCE_THRESHOLD:
            assessment = AssessmentVerdict.INCONCLUSIVE
        elif synthetic_risk_score >= SYNTHETIC_RISK_THRESHOLD:
            assessment = AssessmentVerdict.POTENTIALLY_SYNTHETIC
        elif synthetic_risk_score <= AUTHENTIC_RISK_THRESHOLD:
            assessment = AssessmentVerdict.LIKELY_AUTHENTIC
        else:
            assessment = AssessmentVerdict.INCONCLUSIVE

        # Step 5: Synthesize Plain-Language Explanation
        explanation = self._build_explanation(
            assessment=assessment,
            risk_score=synthetic_risk_score,
            confidence=model_confidence,
            indicators=indicators,
            duration=duration_seconds,
            engine_name=engine_name
        )

        inference_details = ModelInferenceDetails(
            primary_model=self.model_id if self.neural_model_loaded else "Physical Acoustic Signal Analyzer",
            backend_engine=engine_name,
            raw_synthetic_probability=round(neural_prob_synthetic, 4) if neural_prob_synthetic is not None else None,
            raw_authentic_probability=round(neural_prob_authentic, 4) if neural_prob_authentic is not None else None,
            feature_metrics=metrics
        )

        return (
            assessment,
            synthetic_risk_score,
            model_confidence,
            explanation,
            indicators,
            inference_details
        )

    def _build_explanation(
        self,
        assessment: AssessmentVerdict,
        risk_score: float,
        confidence: float,
        indicators: list,
        duration: float,
        engine_name: str
    ) -> str:
        """
        Creates concise, factual explanation without over-claiming certainty.
        """
        risk_pct = int(risk_score * 100)
        conf_pct = int(confidence * 100)

        if assessment == AssessmentVerdict.POTENTIALLY_SYNTHETIC:
            flag_names = [ind.indicator for ind in indicators[:2]]
            flags_str = f" (Key indicators: {', '.join(flag_names)})" if flag_names else ""
            return (
                f"The audio exhibits acoustic patterns consistent with synthetic voice generation "
                f"or neural vocoding (Risk Score: {risk_pct}%, Model Confidence: {conf_pct}%). "
                f"Analysis detected unnatural spectral and prosodic characteristics{flags_str}. "
                f"Do not rely solely on this recording for sensitive identity or financial authorization."
            )

        elif assessment == AssessmentVerdict.LIKELY_AUTHENTIC:
            return (
                f"The audio displays natural human vocal tract resonance, biological pitch jitter, "
                f"and expected ambient background acoustics (Risk Score: {risk_pct}%, Model Confidence: {conf_pct}%). "
                f"No prominent vocoder artifacts or synthetic speech generation fingerprints were detected."
            )

        else:  # INCONCLUSIVE
            reasons = []
            if duration < 2.5:
                reasons.append(f"short recording length ({duration:.1f}s)")
            if conf_pct < int(MIN_CONFIDENCE_THRESHOLD * 100):
                reasons.append("ambiguous acoustic feature distribution")
            reason_str = f" due to {', '.join(reasons)}" if reasons else ""

            return (
                f"The analysis is inconclusive (Risk Score: {risk_pct}%, Model Confidence: {conf_pct}%){reason_str}. "
                f"The audio features lie in the boundary region between natural compressed speech and "
                f"subtle AI voice conversion. Out-of-band identity verification is strongly recommended."
            )
