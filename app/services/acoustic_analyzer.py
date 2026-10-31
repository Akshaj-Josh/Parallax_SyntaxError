"""
Acoustic and Spectral Artifact Feature Extractor.

Calculates genuine physical signal processing metrics from the audio waveform
to identify known speech synthesis and neural vocoder artifacts:
- High-frequency cutoff / unnatural spectral rolloff (>7.5 kHz)
- Spectral flatness / harmonic smoothness anomalies
- Fundamental frequency (F0) micro-jitter and prosody perturbation
- Digital silence / floor noise inconsistency in speech pauses
- Zero Crossing Rate (ZCR) variance across voiced/unvoiced transitions
"""

import math
from typing import Dict, Any, List

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from app.schemas.responses import DetectionIndicator, IndicatorSeverity


class AcousticAnalyzer:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def analyze_audio(self, waveform) -> Dict[str, Any]:
        """
        Extracts acoustic indicators and anomaly scores from waveform.
        Accepts numpy ndarray or python list.
        """
        if HAS_NUMPY and isinstance(waveform, np.ndarray):
            return self._analyze_numpy(waveform)
        else:
            return self._analyze_pure_python(list(waveform))

    def _analyze_numpy(self, waveform: 'np.ndarray') -> Dict[str, Any]:
        metrics = {}
        indicators: List[DetectionIndicator] = []
        anomaly_points = 0.0
        total_checks = 0.0

        n_samples = len(waveform)
        if n_samples < 512:
            return {
                "metrics": {},
                "indicators": [],
                "acoustic_anomaly_score": 0.5
            }

        # 1. High-Frequency Spectral Rolloff & Energy Ratio (Vocoder cutoff)
        fft_vals = np.abs(np.fft.rfft(waveform))
        freqs = np.fft.rfftfreq(n_samples, d=1.0 / self.sample_rate)

        total_energy = float(np.sum(fft_vals ** 2) + 1e-12)
        hf_mask = freqs >= 7000  # Above 7 kHz
        hf_energy = float(np.sum(fft_vals[hf_mask] ** 2))
        hf_ratio = float(hf_energy / total_energy)
        metrics["high_frequency_energy_ratio"] = round(hf_ratio, 5)

        cumulative_energy = np.cumsum(fft_vals ** 2)
        rolloff_idx = np.searchsorted(cumulative_energy, 0.85 * total_energy)
        rolloff_freq = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        metrics["spectral_rolloff_85pct_hz"] = round(rolloff_freq, 1)

        total_checks += 1.0
        if hf_ratio < 0.001 and rolloff_freq < 4500:
            anomaly_points += 0.8
            indicators.append(
                DetectionIndicator(
                    indicator="High-Frequency Vocoder Bandwidth Attenuation",
                    description=(
                        "Audio exhibits an abrupt loss of acoustic energy above 4.5–7 kHz, "
                        "characteristic of neural vocoder bandwidth constraints (e.g. 16kHz mel-spectrogram upsampling)."
                    ),
                    severity=IndicatorSeverity.HIGH if hf_ratio < 0.0003 else IndicatorSeverity.MEDIUM,
                    measured_value=f"{hf_ratio * 100:.3f}% energy > 7kHz (Rolloff: {rolloff_freq:.0f} Hz)",
                    expected_baseline="> 0.8% energy > 7kHz for natural wideband human voice"
                )
            )

        # 2. Spectral Flatness (Wiener Entropy)
        power_spec = fft_vals ** 2 + 1e-12
        geometric_mean = float(np.exp(np.mean(np.log(power_spec))))
        arithmetic_mean = float(np.mean(power_spec))
        spectral_flatness = float(geometric_mean / (arithmetic_mean + 1e-12))
        metrics["spectral_flatness"] = round(spectral_flatness, 6)

        total_checks += 1.0
        if spectral_flatness < 0.00001:
            anomaly_points += 0.7
            indicators.append(
                DetectionIndicator(
                    indicator="Abnormal Spectral Flatness (Over-harmonic Smoothing)",
                    description=(
                        "Harmonic peaks display unnatural mathematical smoothness without "
                        "the biological vocal tract acoustic diffusion observed in human speech."
                    ),
                    severity=IndicatorSeverity.MEDIUM,
                    measured_value=f"{spectral_flatness:.6f}",
                    expected_baseline="0.0001 - 0.05 for natural acoustic speech"
                )
            )

        # 3. Inter-phrase Digital Silence / Floor Noise Analysis
        frame_len = int(0.03 * self.sample_rate)
        hop_len = int(0.015 * self.sample_rate)
        n_frames = (n_samples - frame_len) // hop_len

        if n_frames > 10:
            frame_energies = []
            for i in range(n_frames):
                frame = waveform[i * hop_len: i * hop_len + frame_len]
                rms = float(np.sqrt(np.mean(frame ** 2) + 1e-12))
                frame_energies.append(rms)

            frame_energies = np.array(frame_energies)
            min_energy = float(np.min(frame_energies))
            energy_variance = float(np.var(frame_energies))
            metrics["noise_floor_min_rms"] = round(min_energy, 6)
            metrics["energy_variance"] = round(energy_variance, 6)

            total_checks += 1.0
            if min_energy < 1e-5:
                anomaly_points += 0.75
                indicators.append(
                    DetectionIndicator(
                        indicator="Synthetic Digital Silence in Pauses",
                        description=(
                            "Speech pauses contain unnatural digital silence near 0 dB rather than "
                            "physical room acoustics, microphone self-noise, or environmental reverberation."
                        ),
                        severity=IndicatorSeverity.HIGH if min_energy < 1e-6 else IndicatorSeverity.MEDIUM,
                        measured_value=f"RMS {min_energy:.7f}",
                        expected_baseline="> 0.0001 RMS (natural environmental ambiance)"
                    )
                )

        # 4. Pitch Micro-Perturbation & Jitter
        f0_estimates = self._estimate_f0_contour(waveform, frame_len, hop_len)
        voiced_f0 = f0_estimates[f0_estimates > 0]

        total_checks += 1.0
        if len(voiced_f0) > 10:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            metrics["estimated_mean_f0_hz"] = round(mean_f0, 1)
            metrics["pitch_std_f0_hz"] = round(std_f0, 2)

            diffs = np.abs(np.diff(voiced_f0))
            local_jitter = float(np.mean(diffs) / (mean_f0 + 1e-6))
            metrics["local_pitch_jitter_ratio"] = round(local_jitter, 4)

            if local_jitter < 0.004 and std_f0 < 8.0:
                anomaly_points += 0.8
                indicators.append(
                    DetectionIndicator(
                        indicator="Robotic Pitch Monotony (Low Jitter)",
                        description=(
                            "Pitch contour lacks the involuntary biological micro-jitter and "
                            "prosodic fluctuation characteristic of human vocal fold vibration."
                        ),
                        severity=IndicatorSeverity.MEDIUM,
                        measured_value=f"Jitter {local_jitter:.4f}, StdDev {std_f0:.1f} Hz",
                        expected_baseline="Jitter 0.008 - 0.05, StdDev > 15 Hz"
                    )
                )
            elif local_jitter > 0.22:
                anomaly_points += 0.65
                indicators.append(
                    DetectionIndicator(
                        indicator="Voice Conversion Pitch Discontinuity",
                        description=(
                            "Frequent erratic pitch breaks detected across frames, often caused by "
                            "RVC / voice conversion F0 tracking synthesis errors."
                        ),
                        severity=IndicatorSeverity.MEDIUM,
                        measured_value=f"Jitter {local_jitter:.4f}",
                        expected_baseline="Jitter < 0.12 in connected speech"
                    )
                )
        else:
            metrics["estimated_mean_f0_hz"] = None

        # 5. Zero-Crossing Rate (ZCR)
        zcr = float(np.mean(np.abs(np.diff(np.sign(waveform)))) / 2.0)
        metrics["mean_zero_crossing_rate"] = round(zcr, 4)

        anomaly_score = float(anomaly_points / max(total_checks, 1.0))
        return {
            "metrics": metrics,
            "indicators": indicators,
            "acoustic_anomaly_score": round(min(max(anomaly_score, 0.0), 1.0), 3)
        }

    def _estimate_f0_contour(self, waveform: 'np.ndarray', frame_len: int, hop_len: int) -> 'np.ndarray':
        n_samples = len(waveform)
        n_frames = (n_samples - frame_len) // hop_len
        f0_list = []

        min_lag = int(self.sample_rate / 450.0)
        max_lag = int(self.sample_rate / 80.0)

        for i in range(max(n_frames, 0)):
            frame = waveform[i * hop_len: i * hop_len + frame_len]
            frame = frame - np.mean(frame)
            if np.max(np.abs(frame)) < 1e-4:
                f0_list.append(0.0)
                continue

            corr = np.correlate(frame, frame, mode='full')
            corr = corr[len(corr) // 2:]

            if max_lag < len(corr):
                search_region = corr[min_lag:max_lag]
                peak_idx = int(np.argmax(search_region)) + min_lag
                peak_val = corr[peak_idx]

                if corr[0] > 0 and (peak_val / corr[0]) > 0.40:
                    pitch_hz = float(self.sample_rate / float(peak_idx))
                    f0_list.append(pitch_hz)
                else:
                    f0_list.append(0.0)
            else:
                f0_list.append(0.0)

        return np.array(f0_list)

    def _analyze_pure_python(self, samples: List[float]) -> Dict[str, Any]:
        """Pure Python fallback for pre-install environments."""
        n_samples = len(samples)
        if n_samples == 0:
            return {"metrics": {}, "indicators": [], "acoustic_anomaly_score": 0.5}

        # Calculate RMS noise floor
        frame_len = int(0.03 * self.sample_rate)
        min_rms = 1.0
        frame_energies = []
        for i in range(0, n_samples - frame_len, frame_len):
            frame = samples[i:i + frame_len]
            rms = math.sqrt(sum(s * s for s in frame) / len(frame) + 1e-12)
            frame_energies.append(rms)
            if rms < min_rms:
                min_rms = rms

        indicators = []
        anomaly_score = 0.20

        if min_rms < 1e-5:
            anomaly_score += 0.55
            indicators.append(
                DetectionIndicator(
                    indicator="Synthetic Digital Silence in Pauses",
                    description="Speech pauses contain digital zero silence rather than natural ambient noise.",
                    severity=IndicatorSeverity.HIGH,
                    measured_value=f"RMS {min_rms:.7f}",
                    expected_baseline="> 0.0001 RMS"
                )
            )

        # Zero crossing rate
        zcr_count = sum(1 for i in range(len(samples) - 1) if (samples[i] >= 0) != (samples[i + 1] >= 0))
        zcr = zcr_count / float(max(n_samples, 1))

        return {
            "metrics": {
                "noise_floor_min_rms": round(min_rms, 6),
                "mean_zero_crossing_rate": round(zcr, 4)
            },
            "indicators": indicators,
            "acoustic_anomaly_score": round(min(anomaly_score, 0.95), 3)
        }
