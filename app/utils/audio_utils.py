"""
Audio loading, format conversion, and normalization utilities.
Converts any incoming audio (WAV, MP3, M4A) to 16 kHz mono float32 for model inference.
"""

import os
import wave
import struct
from typing import Tuple, Dict, Any, Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from fastapi import HTTPException, status
except ImportError:
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class status:
        HTTP_400_BAD_REQUEST = 400
        HTTP_413_REQUEST_ENTITY_TOO_LARGE = 413
        HTTP_415_UNSUPPORTED_MEDIA_TYPE = 415
        HTTP_422_UNPROCESSABLE_ENTITY = 422

from app.config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    MIN_AUDIO_DURATION_SECONDS,
    MAX_AUDIO_DURATION_SECONDS
)


def get_audio_info(file_path: str) -> Dict[str, Any]:
    """
    Extracts basic file metadata from an audio file.
    """
    file_size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    # Try standard library wave first for WAV files
    if ext == ".wav":
        try:
            with wave.open(file_path, "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / float(sample_rate) if sample_rate > 0 else 0.0
                return {
                    "duration_seconds": round(duration, 3),
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "file_size_bytes": file_size,
                    "format": "WAV"
                }
        except Exception:
            pass

    # Try soundfile or librosa if available
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return {
            "duration_seconds": round(info.duration, 3),
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "file_size_bytes": file_size,
            "format": info.format
        }
    except Exception:
        pass

    try:
        import librosa
        dur = librosa.get_duration(path=file_path)
        return {
            "duration_seconds": round(float(dur), 3),
            "sample_rate": AUDIO_SAMPLE_RATE,
            "channels": 1,
            "file_size_bytes": file_size,
            "format": ext.replace(".", "").upper()
        }
    except Exception:
        pass

    # Basic fallback estimation
    return {
        "duration_seconds": 0.0,
        "sample_rate": AUDIO_SAMPLE_RATE,
        "channels": AUDIO_CHANNELS,
        "file_size_bytes": file_size,
        "format": ext.replace(".", "").upper()
    }


def load_and_preprocess_audio(
    file_path: str,
    target_sr: int = AUDIO_SAMPLE_RATE,
    max_duration: float = MAX_AUDIO_DURATION_SECONDS
) -> Tuple[np.ndarray, int, float]:
    """
    Loads audio, resamples to target_sr (16,000 Hz), converts to mono, and returns float32 array in [-1.0, 1.0].
    Returns: (waveform_numpy_array, sample_rate, duration_seconds)
    """
    waveform: Optional[np.ndarray] = None
    sr: int = target_sr
    duration: float = 0.0

    # Method 1: Librosa (Preferred for multi-format decoding & resampling)
    try:
        import librosa
        y, orig_sr = librosa.load(file_path, sr=target_sr, mono=True)
        waveform = y.astype(np.float32)
        sr = target_sr
        duration = len(waveform) / float(sr)
    except Exception as e_lib:
        # Method 2: Soundfile
        try:
            import soundfile as sf
            data, orig_sr = sf.read(file_path, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)  # Convert to mono
            if orig_sr != target_sr:
                # Basic linear interpolation resample
                num_samples = int(len(data) * target_sr / orig_sr)
                data = np.interp(
                    np.linspace(0, len(data), num_samples, endpoint=False),
                    np.arange(len(data)),
                    data
                ).astype(np.float32)
            waveform = data
            sr = target_sr
            duration = len(waveform) / float(sr)
        except Exception as e_sf:
            # Method 3: Standard library wave (Native Python fallback for WAV)
            try:
                with wave.open(file_path, "rb") as wf:
                    orig_sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    raw_data = wf.readframes(n_frames)

                    if sampwidth == 2:
                        dtype = np.int16
                        norm = 32768.0
                    elif sampwidth == 1:
                        dtype = np.uint8
                        norm = 128.0
                    elif sampwidth == 4:
                        dtype = np.int32
                        norm = 2147483648.0
                    else:
                        raise ValueError(f"Unsupported sample width: {sampwidth}")

                    audio_arr = np.frombuffer(raw_data, dtype=dtype).astype(np.float32)
                    if sampwidth == 1:
                        audio_arr = (audio_arr - 128.0) / norm
                    else:
                        audio_arr = audio_arr / norm

                    if n_channels > 1:
                        audio_arr = audio_arr.reshape(-1, n_channels).mean(axis=1)

                    if orig_sr != target_sr:
                        num_samples = int(len(audio_arr) * target_sr / orig_sr)
                        audio_arr = np.interp(
                            np.linspace(0, len(audio_arr), num_samples, endpoint=False),
                            np.arange(len(audio_arr)),
                            audio_arr
                        ).astype(np.float32)

                    waveform = audio_arr
                    sr = target_sr
                    duration = len(waveform) / float(sr)
            except Exception as e_wave:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=(
                        f"Failed to decode audio file. Ensure the file is not corrupted. "
                        f"Internal errors: {str(e_lib)[:80]}, {str(e_wave)[:80]}"
                    )
                )

    if waveform is None or len(waveform) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The audio file contains no usable audio samples."
        )

    if duration < MIN_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Audio duration ({duration:.2f}s) is too short for reliable deepfake detection. "
                f"Minimum required is {MIN_AUDIO_DURATION_SECONDS}s."
            )
        )

    if duration > max_duration:
        # Cap to max_duration for latency and resource protection
        max_samples = int(max_duration * target_sr)
        waveform = waveform[:max_samples]
        duration = max_duration

    # Normalize amplitude to prevent volume disparities from skewing features
    max_val = np.max(np.abs(waveform))
    if max_val > 1e-4:
        waveform = waveform / max_val * 0.95

    return waveform, sr, duration
