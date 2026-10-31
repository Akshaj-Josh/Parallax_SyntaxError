/**
 * TRUSTLENS - React Media Analysis Widget
 * 
 * Embeds alongside your existing Phishing Detection results in React.
 * Displays:
 * 1. Audio / Voice Note file uploader (WAV, MP3, M4A)
 * 2. Side-by-side threat comparison (Phishing Score vs Synthetic Audio Score)
 * 3. Specific acoustic anomaly indicators
 * 4. Actionable out-of-band verification steps
 */

import React, { useState } from 'react';

export default function TrustlensMediaWidget({ existingPhishingScore = null }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const API_ENDPOINT = "http://localhost:8001/api/media/analyze-audio";

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setResult(null);
      setError(null);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(API_ENDPOINT, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server returned ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.card}>
      <div style={styles.header}>
        <div style={styles.titleRow}>
          <span style={styles.badgeIcon}>🎙️</span>
          <div>
            <h3 style={styles.heading}>Voice Deepfake & Synthetic Audio Inspection</h3>
            <p style={styles.subtext}>Analyze voicemail, voice memos, or call clips for AI voice cloning.</p>
          </div>
        </div>
      </div>

      {/* Upload Form */}
      <div style={styles.uploadArea}>
        <input
          type="file"
          accept=".wav,.mp3,.m4a,audio/*"
          onChange={handleFileChange}
          style={styles.fileInput}
          id="trustlens-audio-input"
        />
        <label htmlFor="trustlens-audio-input" style={styles.uploadLabel}>
          {selectedFile ? `Selected: ${selectedFile.name}` : "Choose Audio Recording (WAV, MP3, M4A)"}
        </label>

        <button
          onClick={handleAnalyze}
          disabled={!selectedFile || loading}
          style={styles.analyzeBtn}
        >
          {loading ? "Analyzing Acoustics..." : "Scan for AI Voice Cloning"}
        </button>
      </div>

      {error && <div style={styles.errorBox}>⚠️ {error}</div>}

      {/* Analysis Results Display */}
      {result && (
        <div style={styles.resultsContainer}>
          {/* Side-by-side Score Matrix */}
          <div style={styles.scoreMatrix}>
            {/* Existing Phishing Score (if provided by your teammate's model) */}
            {existingPhishingScore !== null && (
              <div style={styles.scoreCard}>
                <span style={styles.scoreLabel}>Text Phishing Threat</span>
                <span style={styles.scoreValue}>{Math.round(existingPhishingScore * 100)}%</span>
              </div>
            )}

            {/* Deepfake Risk Score */}
            <div style={{
              ...styles.scoreCard,
              borderColor: result.synthetic_risk_score > 0.6 ? '#ef4444' : '#10b981'
            }}>
              <span style={styles.scoreLabel}>Synthetic Voice Risk</span>
              <span style={{
                ...styles.scoreValue,
                color: result.synthetic_risk_score > 0.6 ? '#ef4444' : '#10b981'
              }}>
                {Math.round(result.synthetic_risk_score * 100)}%
              </span>
              <span style={styles.confidenceLabel}>
                Confidence: {Math.round(result.model_confidence * 100)}%
              </span>
            </div>
          </div>

          {/* Verdict Banner */}
          <div style={{
            ...styles.verdictBanner,
            backgroundColor: result.overall_assessment === 'potentially_synthetic'
              ? 'rgba(239, 68, 68, 0.15)'
              : 'rgba(16, 185, 129, 0.15)'
          }}>
            <strong>Assessment: </strong> {result.overall_assessment.replace('_', ' ').toUpperCase()}
            <p style={styles.explanationText}>{result.explanation}</p>
          </div>

          {/* Acoustic Indicators */}
          {result.detection_indicators && result.detection_indicators.length > 0 && (
            <div style={styles.indicatorSection}>
              <h4 style={styles.subHeading}>Detected Acoustic Anomalies</h4>
              {result.detection_indicators.map((ind, i) => (
                <div key={i} style={styles.indicatorItem}>
                  <strong>{ind.indicator}</strong> ({ind.severity} severity)
                  <p style={styles.indicatorDesc}>{ind.description}</p>
                  {ind.measured_value && <span style={styles.indicatorVal}>Measured: {ind.measured_value}</span>}
                </div>
              ))}
            </div>
          )}

          {/* Verification Recommendations */}
          {result.verification_guidance && (
            <div style={styles.guidanceBox}>
              <h4 style={styles.guidanceHeading}>🛡️ Out-of-Band Verification Next Steps</h4>
              <ul style={styles.guidanceList}>
                {result.verification_guidance.immediate_actions.map((act, idx) => (
                  <li key={idx}>{act}</li>
                ))}
              </ul>
              <div style={styles.avoidBox}>
                <strong>What to strictly avoid:</strong>
                <ul>
                  {result.verification_guidance.what_to_avoid.map((av, idx) => (
                    <li key={idx}>{av}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const styles = {
  card: {
    backgroundColor: '#162032',
    border: '1px solid #24344d',
    borderRadius: '12px',
    padding: '24px',
    color: '#f3f4f6',
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    marginTop: '20px'
  },
  header: { marginBottom: '16px' },
  titleRow: { display: 'flex', alignItems: 'center', gap: '12px' },
  badgeIcon: { fontSize: '24px' },
  heading: { margin: 0, fontSize: '18px', fontWeight: '600' },
  subtext: { margin: '4px 0 0', fontSize: '13px', color: '#9ca3af' },
  uploadArea: { display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap', marginBottom: '16px' },
  fileInput: { display: 'none' },
  uploadLabel: {
    padding: '10px 16px',
    backgroundColor: '#0b1320',
    border: '1px solid #24344d',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '13px'
  },
  analyzeBtn: {
    padding: '10px 20px',
    backgroundColor: '#2563eb',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontWeight: '600',
    fontSize: '14px'
  },
  errorBox: {
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    border: '1px solid #ef4444',
    color: '#fca5a5',
    padding: '12px',
    borderRadius: '8px',
    fontSize: '13px',
    marginBottom: '16px'
  },
  resultsContainer: { marginTop: '20px' },
  scoreMatrix: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px', marginBottom: '16px' },
  scoreCard: {
    backgroundColor: '#0b1320',
    border: '1px solid #24344d',
    borderRadius: '8px',
    padding: '16px',
    textAlign: 'center'
  },
  scoreLabel: { display: 'block', fontSize: '12px', color: '#9ca3af', textTransform: 'uppercase' },
  scoreValue: { display: 'block', fontSize: '28px', fontWeight: '800', margin: '4px 0' },
  confidenceLabel: { fontSize: '11px', color: '#6b7280' },
  verdictBanner: { padding: '16px', borderRadius: '8px', marginBottom: '16px' },
  explanationText: { fontSize: '13px', marginTop: '6px', lineHeight: '1.5' },
  indicatorSection: { marginBottom: '16px' },
  subHeading: { fontSize: '14px', marginBottom: '8px' },
  indicatorItem: {
    backgroundColor: '#0b1320',
    border: '1px solid #24344d',
    padding: '10px 14px',
    borderRadius: '6px',
    marginBottom: '8px',
    fontSize: '13px'
  },
  indicatorDesc: { margin: '4px 0', fontSize: '12px', color: '#9ca3af' },
  indicatorVal: { fontSize: '11px', color: '#60a5fa' },
  guidanceBox: { backgroundColor: '#0f172a', border: '1px solid #1e293b', borderRadius: '8px', padding: '16px' },
  guidanceHeading: { margin: '0 0 10px', fontSize: '14px', color: '#10b981' },
  guidanceList: { margin: '0 0 12px', paddingLeft: '20px', fontSize: '13px', lineHeight: '1.6' },
  avoidBox: { backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.3)', padding: '12px', borderRadius: '6px', fontSize: '13px' }
};
