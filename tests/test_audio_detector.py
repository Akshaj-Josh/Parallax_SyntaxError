"""
Unit tests for AudioDeepfakeDetector and AcousticAnalyzer.
Verifies signal processing metrics, threshold calibration, and error resilience.
"""

import unittest
import math
import random

try:
    import numpy as np
except ImportError:
    np = None

from app.services.acoustic_analyzer import AcousticAnalyzer
from app.services.audio_detector import AudioDeepfakeDetector
from app.schemas.responses import AssessmentVerdict


class TestAudioDeepfakeDetector(unittest.TestCase):
    def setUp(self):
        self.sample_rate = 16000
        self.analyzer = AcousticAnalyzer(sample_rate=self.sample_rate)
        self.detector = AudioDeepfakeDetector()

    def test_synthetic_vocoder_anomaly_detection(self):
        """
        Tests that an audio signal with typical vocoder artifacts
        (lowpass cutoff at 3.5 kHz, flat pitch, zero noise floor)
        triggers acoustic anomaly indicators and high risk.
        """
        duration = 3.0
        n_samples = int(self.sample_rate * duration)
        f0 = 150.0

        if np is not None:
            t = np.linspace(0, duration, n_samples, endpoint=False)
            signal = 0.6 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            signal[int(1.0 * self.sample_rate): int(1.5 * self.sample_rate)] = 0.0
        else:
            signal = []
            for i in range(n_samples):
                t_sec = i / float(self.sample_rate)
                if 1.0 <= t_sec <= 1.5:
                    signal.append(0.0)
                else:
                    val = 0.6 * math.sin(2 * math.pi * f0 * t_sec) + 0.3 * math.sin(2 * math.pi * 2 * f0 * t_sec)
                    signal.append(val)

        res = self.analyzer.analyze_audio(signal)
        indicators = res["indicators"]

        indicator_names = [ind.indicator for ind in indicators]
        self.assertTrue(
            any("Silence" in name for name in indicator_names),
            f"Expected digital silence indicator, got: {indicator_names}"
        )

        verdict, risk, conf, exp, inds, details = self.detector.analyze_audio(signal, duration)
        self.assertGreaterEqual(risk, 0.40)
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIsInstance(verdict, AssessmentVerdict)

    def test_authentic_speech_simulation_baseline(self):
        """
        Tests that a signal simulating natural speech with micro-jitter
        and natural background noise produces a lower risk score.
        """
        duration = 3.5
        n_samples = int(self.sample_rate * duration)

        if np is not None:
            t = np.linspace(0, duration, n_samples, endpoint=False)
            f0 = 140.0 + 8.0 * np.sin(2 * np.pi * 1.5 * t) + np.random.normal(0, 1.0, len(t))
            phase = 2 * np.pi * np.cumsum(f0) / self.sample_rate
            natural_speech = 0.50 * np.sin(phase) + 0.30 * np.sin(2 * phase)
            audio = natural_speech * 0.7 + np.random.normal(0, 0.015, len(t))
        else:
            audio = []
            phase = 0.0
            for i in range(n_samples):
                t_sec = i / float(self.sample_rate)
                f0 = 140.0 + 8.0 * math.sin(2 * math.pi * 1.5 * t_sec) + random.gauss(0, 1.0)
                phase += 2 * math.pi * f0 / self.sample_rate
                natural_speech = 0.50 * math.sin(phase) + 0.30 * math.sin(2 * phase)
                audio.append(natural_speech * 0.7 + random.gauss(0, 0.015))

        verdict, risk, conf, exp, inds, details = self.detector.analyze_audio(audio, duration)
        self.assertLess(risk, 0.65)
        self.assertGreater(conf, 0.3)

    def test_confidence_and_risk_separation(self):
        """
        Verifies that risk score and model confidence are strictly distinct fields.
        """
        n_samples = int(self.sample_rate * 2.0)
        if np is not None:
            t = np.linspace(0, 2.0, n_samples, endpoint=False)
            audio = 0.4 * np.sin(2 * np.pi * 300 * t)
        else:
            audio = [0.4 * math.sin(2 * math.pi * 300 * (i / self.sample_rate)) for i in range(n_samples)]

        verdict, risk, conf, exp, inds, details = self.detector.analyze_audio(audio, 2.0)
        self.assertNotEqual(risk, conf, "Risk score and model confidence must be distinct metrics.")


if __name__ == "__main__":
    unittest.main()
