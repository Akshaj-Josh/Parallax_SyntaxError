"""
Unit tests for VerificationAdvisor.
Ensures appropriate security advice is rendered for all risk tiers.
"""

import unittest
from app.services.verification_advisor import VerificationAdvisor
from app.schemas.responses import AssessmentVerdict


class TestVerificationAdvisor(unittest.TestCase):
    def test_synthetic_guidance_structure(self):
        guidance = VerificationAdvisor.get_guidance(
            verdict=AssessmentVerdict.POTENTIALLY_SYNTHETIC,
            risk_score=0.88,
            media_type="audio"
        )
        self.assertIn("High-Risk", guidance.category)
        self.assertTrue(len(guidance.immediate_actions) >= 3)
        self.assertTrue(len(guidance.what_to_avoid) >= 3)
        self.assertTrue(any("Out-of-Band" in act for act in guidance.immediate_actions))
        self.assertTrue(any("DO NOT" in avoid for avoid in guidance.what_to_avoid))

    def test_inconclusive_guidance_structure(self):
        guidance = VerificationAdvisor.get_guidance(
            verdict=AssessmentVerdict.INCONCLUSIVE,
            risk_score=0.50,
            media_type="audio"
        )
        self.assertIn("Inconclusive", guidance.category)
        self.assertTrue(any("secondary" in act.lower() for act in guidance.immediate_actions))

    def test_authentic_guidance_structure(self):
        guidance = VerificationAdvisor.get_guidance(
            verdict=AssessmentVerdict.LIKELY_AUTHENTIC,
            risk_score=0.18,
            media_type="audio"
        )
        self.assertIn("Standard", guidance.category)
        self.assertTrue(any("Standard" in act for act in guidance.immediate_actions))


if __name__ == "__main__":
    unittest.main()
