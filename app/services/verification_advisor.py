"""
TRUSTLENS Verification Guidance Advisor.

Generates contextual, actionable security recommendations to counter
voice cloning, CEO fraud, synthetic voicemail attacks, and family emergency scams.
"""

from app.schemas.responses import AssessmentVerdict, VerificationRecommendation


class VerificationAdvisor:
    """
    Generates tailored, safe verification guidance based on media risk assessment.
    """

    @staticmethod
    def get_guidance(
        verdict: AssessmentVerdict,
        risk_score: float,
        media_type: str = "audio"
    ) -> VerificationRecommendation:
        is_audio = "audio" in media_type.lower()
        media_label = "audio recording / voice message" if is_audio else "video recording"

        if verdict == AssessmentVerdict.POTENTIALLY_SYNTHETIC:
            return VerificationRecommendation(
                category="High-Risk Synthetic Voice / Deepfake Impersonation Advisory",
                immediate_actions=[
                    "Initiate Out-of-Band Callback: Call the claimed speaker directly using a pre-saved phone number from your personal contact list or official company directory. NEVER dial a phone number provided inside the message.",
                    "Require Dual Authorization: If the message requests an emergency wire transfer, gift card purchase, crypto transaction, or urgent vendor payment, enforce standard two-person internal approval.",
                    "Deploy Challenge-Response Verification: If in a live call or voicemail, ask a specific question whose answer is known only to the real individual and not publicly accessible on social media.",
                    "Check for Spoofed Caller ID: Remember that phone numbers and caller ID displays can be trivially spoofed alongside cloned voices."
                ],
                what_to_avoid=[
                    f"❌ DO NOT authorize any financial wire, gift card purchase, or fund transfer based on this {media_label}.",
                    f"❌ DO NOT share one-time passwords (OTPs), account PINs, credentials, or employee logins.",
                    f"❌ DO NOT call back any telephone number mentioned in the audio or video body.",
                    f"❌ DO NOT assume voice familiarity is proof of identity—AI voice clones need as little as 3 seconds of reference audio."
                ],
                out_of_band_protocol=(
                    "Standard Out-of-Band (OOB) Protocol: Always switch to a physically separate, pre-authenticated "
                    "communication channel (e.g., in-person confirmation, known internal corporate Slack/Teams channel, "
                    "or landline phone) before taking action on urgent or financial requests."
                )
            )

        elif verdict == AssessmentVerdict.INCONCLUSIVE:
            return VerificationRecommendation(
                category="Precautionary Verification Advisory (Inconclusive Media)",
                immediate_actions=[
                    "Request Higher Quality Audio: Background noise or low bitrate prevented definitive classification. If feasible, request a live two-way phone call.",
                    "Verify Sender Authenticity Out-of-Band: Confirm the request via a secondary trusted communication channel before proceeding.",
                    "Scrutinize Uncharacteristic Urgency: High urgency and emotional pressure are the primary psychological manipulation levers in social engineering attacks."
                ],
                what_to_avoid=[
                    "❌ DO NOT make irreversible financial decisions or grant access permissions without secondary confirmation.",
                    "❌ DO NOT bypass established standard verification procedures due to perceived urgency."
                ],
                out_of_band_protocol=(
                    "Treat inconclusive recordings with healthy skepticism. Contact the alleged sender via a known trusted "
                    "secondary contact channel."
                )
            )

        else:  # LIKELY_AUTHENTIC
            return VerificationRecommendation(
                category="Standard Security Best Practices",
                immediate_actions=[
                    "Maintain Standard Verification: While no synthetic artifacts were detected, continue adhering to standard verification for high-value financial transactions.",
                    "Verify Identity for Unusual Inquiries: Legitimate individuals can be coerced or compromised."
                ],
                what_to_avoid=[
                    "❌ DO NOT bypass two-factor authentication or corporate authorization protocols.",
                    "❌ DO NOT share private credentials or PIN numbers over voice or messaging channels."
                ],
                out_of_band_protocol=(
                    "Even when audio appears authentic, critical financial and credential requests should follow standard "
                    "dual-authorization verification workflows."
                )
            )
