"""
test_calibration.py
-------------------
Unit tests for the calibration layer (ordinal confidence → float score).
"""

import pytest
from unittest.mock import MagicMock
from app.services.calibration import calibrate_extraction, CONFIDENCE_MAP
from app.models.schemas import (
    ConfidenceLevel,
    MedicalExtraction,
    ImageQuality,
    ImageQualityIssues,
    UncertaintySources,
    OutOfDistribution,
    StructuredFinding,
)


def _make_extraction(findings=None) -> MedicalExtraction:
    """Helper: builds a minimal MedicalExtraction for testing."""
    return MedicalExtraction(
        modality="Xray",
        anatomy="Chest",
        image_quality=ImageQuality(
            usable=True,
            issues=ImageQualityIssues(blur=False, low_contrast=False, glare=False, cropped=False),
            reason_if_unusable=None,
        ),
        uncertainty_sources=UncertaintySources(
            image_quality_issues=[],
            visual_evidence_strength=80,
            model_ambiguity=False,
            out_of_distribution=OutOfDistribution(detected=False, reason=None),
        ),
        findings=findings or [],
        retrieval_queries=["chest xray pleural effusion"],
        recommended_pipeline="Chest_Xray",
    )


# ── CONFIDENCE_MAP ────────────────────────────────────────────────────────────

class TestConfidenceMap:
    def test_all_levels_present(self):
        expected = {ConfidenceLevel.very_high, ConfidenceLevel.high, ConfidenceLevel.medium, ConfidenceLevel.low}
        assert set(CONFIDENCE_MAP.keys()) == expected

    def test_scores_are_descending(self):
        scores = [
            CONFIDENCE_MAP[ConfidenceLevel.very_high],
            CONFIDENCE_MAP[ConfidenceLevel.high],
            CONFIDENCE_MAP[ConfidenceLevel.medium],
            CONFIDENCE_MAP[ConfidenceLevel.low],
        ]
        assert scores == sorted(scores, reverse=True)

    def test_scores_in_range(self):
        for score in CONFIDENCE_MAP.values():
            assert 0.0 <= score <= 1.0


# ── calibrate_extraction() ────────────────────────────────────────────────────

class TestCalibrateExtraction:
    def test_no_findings(self):
        result = calibrate_extraction(_make_extraction(findings=[]))
        assert result["findings"] == []
        assert result["modality"] == "Xray"
        assert result["anatomy"] == "Chest"

    def test_single_very_high_finding(self):
        finding = StructuredFinding(
            finding="Pleural Effusion",
            confidence_level=ConfidenceLevel.very_high,
            supporting_evidence=["blunted costophrenic angle"],
        )
        result = calibrate_extraction(_make_extraction(findings=[finding]))
        assert len(result["findings"]) == 1
        f = result["findings"][0]
        assert f["calibrated_score"] == CONFIDENCE_MAP[ConfidenceLevel.very_high]
        assert f["confidence_level"] == "very_high"
        assert "blunted costophrenic angle" in f["supporting_evidence"]

    def test_findings_sorted_by_score_descending(self):
        findings = [
            StructuredFinding(
                finding="Low Confidence Finding",
                confidence_level=ConfidenceLevel.low,
                supporting_evidence=[],
            ),
            StructuredFinding(
                finding="High Confidence Finding",
                confidence_level=ConfidenceLevel.very_high,
                supporting_evidence=[],
            ),
        ]
        result = calibrate_extraction(_make_extraction(findings=findings))
        scores = [f["calibrated_score"] for f in result["findings"]]
        assert scores == sorted(scores, reverse=True)

    def test_retrieval_queries_passed_through(self):
        result = calibrate_extraction(_make_extraction())
        assert result["retrieval_queries"] == ["chest xray pleural effusion"]

    def test_recommended_pipeline_passed_through(self):
        result = calibrate_extraction(_make_extraction())
        assert result["recommended_pipeline"] == "Chest_Xray"

    def test_uncertainty_sources_structure(self):
        result = calibrate_extraction(_make_extraction())
        unc = result["uncertainty_sources"]
        assert "image_quality_issues" in unc
        assert "visual_evidence_strength" in unc
        assert "model_ambiguity" in unc
        assert "out_of_distribution" in unc

    def test_visual_evidence_strength_formatted_as_percent(self):
        result = calibrate_extraction(_make_extraction())
        assert result["uncertainty_sources"]["visual_evidence_strength"] == "80%"
