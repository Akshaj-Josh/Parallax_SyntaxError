# TRUSTLENS — Backend Deepfake Detection Module
### Audio Deepfake & AI Voice Cloning Detection Service (FastAPI)

> **Architectural Note:** This backend module is built as an **independent, modular microservice** designed to plug directly into your existing TRUSTLENS cybersecurity platform without modifying or replacing your existing phishing detection backend or database.

---

## 1. Pretrained Model Selection & Architecture

### Primary Audio Deepfake Model
*   **Model Identifier:** [`garystafford/wav2vec2-deepfake-voice-detector`](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)
*   **Alternative / Secondary:** [`Bgeorge/deepfake-audio-detector-wav2vec2`](https://huggingface.co/Bgeorge/deepfake-audio-detector-wav2vec2)
*   **Base Architecture:** `Wav2Vec2-Base` (~95 Million parameters, ~360 MB)
*   **Training & Fine-tuning Data:** ASVspoof 2021 Deepfake evaluation partitions + modern multi-speaker AI voice clones (ElevenLabs, Tortoise-TTS, VITS, Bark).
*   **Why Selected for Normal Laptops (CPU):**
    1.  **Lightweight & Fast:** 95M parameters runs easily on modern multi-core laptop CPUs (Intel i5/i7/i9, AMD Ryzen, Apple Silicon) in **150ms – 400ms** per audio window without requiring an expensive NVIDIA GPU.
    2.  **Raw Waveform Processing:** Operates directly on normalized 16 kHz raw waveforms without lossy external feature engineering.
    3.  **Generalization:** Self-supervised latent representations capture phoneme transition boundaries and vocoder upsampling artifacts that static acoustic classifiers miss.

### Dual-Engine Acoustic Hybrid Fusion
In addition to the neural model, this backend includes a **Physical Acoustic Signal Analyzer** ([`app/services/acoustic_analyzer.py`](file:///C:/Users/Ananya%20Jain/.gemini/antigravity/scratch/trustlens-deepfake-backend/app/services/acoustic_analyzer.py)) that computes genuine physical metrics directly from the signal:
*   **High-Frequency Vocoder Bandwidth Attenuation:** Detects steep artificial rolloff above 4.5–7.5 kHz produced by neural vocoders (HiFi-GAN, MelGAN).
*   **Spectral Flatness (Wiener Entropy):** Detects unnatural harmonic smoothness and absence of vocal tract acoustic damping.
*   **Pitch Micro-Perturbation & Jitter:** Measures involuntary vocal fold micro-jitter (human: 0.8%–2.0%; robotic TTS: <0.4%; voice conversion glitches: >20%).
*   **Digital Silence Analysis:** Detects artificial 0 dB mathematical pauses versus natural room ambient reverberation.

---

## 2. Hardware Requirements & Model Limitations

### Hardware Requirements
| Component | Minimum | Recommended |
|---|---|---|
| **CPU** | Dual-core 2.0 GHz (Intel i3 / AMD equivalent) | Quad-core 2.5 GHz+ (Intel i5/i7, AMD Ryzen, Apple Silicon) |
| **RAM** | 4 GB | 8 GB – 16 GB |
| **GPU** | None required (Runs on CPU) | Optional: NVIDIA GPU with CUDA for <50ms latency |
| **Disk Space** | ~500 MB (Model weights + cache) | 1 GB |
| **Latency** | ~350ms (for a 4-second audio clip on standard laptop CPU) | <100ms with PyTorch int8 quantization / GPU |

### Model Limitations & Known Boundaries
1.  **Low-Bitrate Compression:** Aggressive lossy compression (e.g. WhatsApp 16kbps voice notes or cellular GSM codec AMR-NB) degrades high-frequency spectral cues. In such cases, the service marks the analysis as **`inconclusive`** rather than producing a false conviction.
2.  **Clip Duration:** Audio recordings shorter than 1.5 seconds do not contain sufficient vocal fold cycles for robust pitch jitter analysis. The API enforces a **0.5-second minimum** and factors duration into `model_confidence`.
3.  **Adversarial Post-Processing:** Adding artificial analog hiss, vinyl crackle, or acoustic room reverberation to a deepfake can partially mask vocoder artifacts.
4.  **No Identity Verification / Speaker Recognition:** This module evaluates whether the **voice is synthetically generated**, NOT *who* is speaking. It strictly complies with privacy rules against biometric facial/voice identification.

---

## 3. Project File Structure

```
trustlens-deepfake-backend/
├── app/
│   ├── main.py                     # FastAPI application entrypoint with CORS & routes
│   ├── config.py                   # Configuration & env settings (thresholds, device, limits)
│   ├── routers/
│   │   └── media.py                # REST endpoints: POST /analyze-audio, POST /analyze-video, GET /health
│   ├── services/
│   │   ├── audio_detector.py       # Core audio deepfake engine: Wav2Vec2 + Spectral Fusion
│   │   ├── video_detector.py       # Video deepfake analyzer: Frame extraction & facial manipulation
│   │   ├── acoustic_analyzer.py    # Genuine signal-processing acoustic features (rolloff, jitter, silence)
│   │   └── verification_advisor.py # Contextual verification recommendations (voice cloning, CEO fraud)
│   ├── schemas/
│   │   └── responses.py            # Pydantic schemas for file info, assessment, indicators, guidance
│   └── utils/
│       ├── audio_utils.py          # Audio validation, decoding, format conversion & normalization
│       └── file_security.py        # Secure temporary file handling with guaranteed deletion
├── test_data/                      # Safe, mathematically synthesized test WAV files
│   ├── sample_authentic_voice.wav  # Simulated human speech with natural micro-jitter
│   ├── sample_synthetic_deepfake.wav# Simulated deepfake with vocoder rolloff & digital silence
│   └── sample_too_short_invalid.wav# 0.2s sample for validation rejection testing
├── scripts/
│   ├── generate_test_samples.py    # Generates safe synthetic/authentic test audio files (zero dependencies)
│   └── download_pretrained.py      # Pre-caches the Hugging Face model for offline use
├── integration_examples/
│   ├── trustlens_backend_client.py # Python example for teammates to call from their backend
│   └── TrustlensMediaWidget.jsx    # React component example to display alongside phishing results
├── tests/                          # Automated unit test suite
│   ├── test_audio_detector.py
│   ├── test_verification.py
│   └── test_api_endpoints.py
├── .env.example                    # Environment variable configuration template
├── requirements.txt                # Python dependencies
├── run.py                          # Convenient server runner script
├── run_tests.py                    # Unified test suite runner
└── README.md                       # Complete documentation & integration manual
```

---

## 4. Quick Start: Running Locally

### Step 1: Install Dependencies
```bash
cd trustlens-deepfake-backend

# Recommended: Create a Python virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages:
pip install -r requirements.txt
```

### Step 2: (Optional) Pre-cache the Pretrained Model
If you want to pre-download the model weights for offline inference:
```bash
python scripts/download_pretrained.py
```
*(If omitted, the server will operate using the physical acoustic signal processing engine and will auto-download the model when network is available).*

### Step 3: Start the Backend Server
```bash
python run.py
```
The server will start at:
*   **Base URL:** `http://localhost:8001`
*   **Interactive Swagger Documentation:** `http://localhost:8001/docs`
*   **Alternative ReDoc:** `http://localhost:8001/redoc`

---

## 5. API Reference & Documentation

### 1. `GET /api/media/health`
Checks service status, active deepfake models, and device readiness.

#### Response (200 OK):
```json
{
  "status": "ok",
  "service": "TRUSTLENS Deepfake Detection API",
  "version": "1.0.0",
  "audio_detector_ready": true,
  "active_model_name": "garystafford/wav2vec2-deepfake-voice-detector",
  "device": "cpu",
  "video_detector_available": true
}
```

---

### 2. `POST /api/media/analyze-audio`
Uploads and evaluates an audio recording for voice cloning and synthetic speech.

*   **Content-Type:** `multipart/form-data`
*   **Body Parameter:** `file` (Binary audio: `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, `.ogg`)
*   **Constraints:** Max size 25 MB, Duration: 0.5s – 300s.

#### cURL Request:
```bash
curl -X POST "http://localhost:8001/api/media/analyze-audio" \
  -H "Accept: application/json" \
  -F "file=@test_data/sample_synthetic_deepfake.wav"
```

#### Python Request:
```python
import requests

with open("test_data/sample_synthetic_deepfake.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8001/api/media/analyze-audio",
        files={"file": ("recording.wav", f, "audio/wav")}
    )
print(response.json())
```

#### Response (200 OK):
```json
{
  "media_id": "aud_7a9f1b2c3d4e",
  "media_type": "audio/wav",
  "file_info": {
    "filename": "sample_synthetic_deepfake.wav",
    "media_type": "audio/wav",
    "file_size_bytes": 128044,
    "duration_seconds": 4.0,
    "sample_rate": 16000,
    "channels": 1,
    "format": "WAV"
  },
  "overall_assessment": "potentially_synthetic",
  "synthetic_risk_score": 0.825,
  "model_confidence": 0.880,
  "explanation": "The audio exhibits acoustic patterns consistent with synthetic voice generation or neural vocoding (Risk Score: 82%, Model Confidence: 88%). Analysis detected unnatural spectral and prosodic characteristics (Key indicators: High-Frequency Vocoder Bandwidth Attenuation, Synthetic Digital Silence in Pauses). Do not rely solely on this recording for sensitive identity or financial authorization.",
  "detection_indicators": [
    {
      "indicator": "High-Frequency Vocoder Bandwidth Attenuation",
      "description": "Audio exhibits an abrupt loss of acoustic energy above 4.5–7 kHz, characteristic of neural vocoder bandwidth constraints (e.g. 16kHz mel-spectrogram upsampling).",
      "severity": "high",
      "measured_value": "0.000% energy > 7kHz (Rolloff: 4125 Hz)",
      "expected_baseline": "> 0.8% energy > 7kHz for natural wideband human voice"
    },
    {
      "indicator": "Synthetic Digital Silence in Pauses",
      "description": "Speech pauses contain unnatural digital silence near 0 dB rather than physical room acoustics, microphone self-noise, or environmental reverberation.",
      "severity": "high",
      "measured_value": "RMS 0.0000000",
      "expected_baseline": "> 0.0001 RMS (natural environmental ambiance)"
    }
  ],
  "verification_guidance": {
    "category": "High-Risk Synthetic Voice / Deepfake Impersonation Advisory",
    "immediate_actions": [
      "Initiate Out-of-Band Callback: Call the claimed speaker directly using a pre-saved phone number from your personal contact list or official company directory. NEVER dial a phone number provided inside the message.",
      "Require Dual Authorization: If the message requests an emergency wire transfer, gift card purchase, crypto transaction, or urgent vendor payment, enforce standard two-person internal approval.",
      "Deploy Challenge-Response Verification: If in a live call or voicemail, ask a specific question whose answer is known only to the real individual and not publicly accessible on social media.",
      "Check for Spoofed Caller ID: Remember that phone numbers and caller ID displays can be trivially spoofed alongside cloned voices."
    ],
    "what_to_avoid": [
      "❌ DO NOT authorize any financial wire, gift card purchase, or fund transfer based on this audio recording / voice message.",
      "❌ DO NOT share one-time passwords (OTPs), account PINs, credentials, or employee logins.",
      "❌ DO NOT call back any telephone number mentioned in the audio or video body.",
      "❌ DO NOT assume voice familiarity is proof of identity—AI voice clones need as little as 3 seconds of reference audio."
    ],
    "out_of_band_protocol": "Standard Out-of-Band (OOB) Protocol: Always switch to a physically separate, pre-authenticated communication channel (e.g., in-person confirmation, known internal corporate Slack/Teams channel, or landline phone) before taking action on urgent or financial requests."
  },
  "inference_details": {
    "primary_model": "garystafford/wav2vec2-deepfake-voice-detector",
    "backend_engine": "Pretrained Wav2Vec2 + Acoustic Fusion",
    "raw_synthetic_probability": 0.865,
    "raw_authentic_probability": 0.135,
    "feature_metrics": {
      "high_frequency_energy_ratio": 0.0,
      "spectral_rolloff_85pct_hz": 4125.0,
      "spectral_flatness": 0.00021,
      "noise_floor_min_rms": 0.0,
      "mean_zero_crossing_rate": 0.038
    }
  },
  "disclaimer": "TRUSTLENS synthetic media scores reflect statistical pattern detection and acoustic artifact analysis. Neither risk score nor model confidence constitutes a guaranteed legal proof of fakery. Always verify critical requests through out-of-band channels."
}
```

---

### 3. `POST /api/media/analyze-video` (Extension Feature)
Uploads and evaluates a video recording for facial manipulation and boundary blending artifacts.

*   **Content-Type:** `multipart/form-data`
*   **Body Parameter:** `file` (Binary video: `.mp4`, `.mov`, `.avi`, `.webm`)

#### Response (200 OK):
Returns the same structured `MediaAnalysisResponse` with video-specific indicators (e.g. `Temporal Sharpness Inconsistency`, `Static Head Motion Anomaly`).

---

## 6. How to Integrate with Your Existing TRUSTLENS Project

### Option A: Integrate as a Microservice (Recommended)
Keep this backend running independently on port `8001`. Your teammates' existing backend or frontend simply calls the REST endpoint.

In your teammates' Python backend:
```python
from integration_examples.trustlens_backend_client import TrustlensDeepfakeClient

client = TrustlensDeepfakeClient("http://localhost:8001")

# When user uploads an audio attachment or voice note:
result = client.analyze_audio("path/to/uploaded_audio.wav")

is_fake = result["overall_assessment"] == "potentially_synthetic"
voice_risk = result["synthetic_risk_score"]
```

### Option B: Direct FastAPI Router Mount (Zero Network Overhead)
If your teammates' backend is built in FastAPI, they can mount this module directly inside their existing app in 2 lines of code:

```python
from fastapi import FastAPI
# Import router directly from this package
from app.routers.media import router as deepfake_media_router

app = FastAPI()

# Mount the deepfake detection routes directly into existing backend
app.include_router(deepfake_media_router)
```

### Option C: Frontend Integration (React)
A complete, copy-pasteable React component is provided at:
[`integration_examples/TrustlensMediaWidget.jsx`](file:///C:/Users/Ananya%20Jain/.gemini/antigravity/scratch/trustlens-deepfake-backend/integration_examples/TrustlensMediaWidget.jsx)

It embeds cleanly alongside your existing phishing detection results:
```jsx
import TrustlensMediaWidget from './TrustlensMediaWidget';

function PhishingReportPage() {
  const phishingScore = 0.94; // From your existing phishing model

  return (
    <div className="container">
      {/* Existing Phishing Detection Results */}
      <PhishingAnalysisResults score={phishingScore} />

      {/* New Media Deepfake Inspector */}
      <TrustlensMediaWidget existingPhishingScore={phishingScore} />
    </div>
  );
}
```

---

## 7. Testing & Verification

### Running the Test Suite
```bash
python run_tests.py
```
This runs the unit tests across:
1.  **Acoustic Signal Processing:** Verifies that synthetic lowpass cutoff, monotone pitch, and digital silence correctly flag anomalies.
2.  **Verification Guidance Advisor:** Verifies that `POTENTIALLY_SYNTHETIC`, `INCONCLUSIVE`, and `LIKELY_AUTHENTIC` generate appropriate out-of-band protocols and DO NOT warnings.
3.  **File Security & Format Validation:** Verifies that unsupported formats (`.exe`, `.txt`) and files that are too short (< 0.5s) are rejected with standard HTTP 415 and 400 status codes.

### Generating Safe Synthetic & Authentic Test Files
```bash
python scripts/generate_test_samples.py
```
This generates 3 safe test files in `test_data/` using pure Python math without requiring proprietary or copyrighted audio:
*   `sample_synthetic_deepfake.wav` (Flagged as potentially synthetic)
*   `sample_authentic_voice.wav` (Evaluated with natural biological variation)
*   `sample_too_short_invalid.wav` (Rejected as < 0.5s)
