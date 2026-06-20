"""
Calibration Layer
-----------------
Converts ordinal confidence labels (from Gemini) into calibrated float scores.

Mapping:
  very_high → 0.90
  high      → 0.75
  medium    → 0.50
  low       → 0.25
"""

from app.models.schemas import ConfidenceLevel, MedicalExtraction


CONFIDENCE_MAP: dict[str, float] = {
    ConfidenceLevel.very_high: 0.90,
    ConfidenceLevel.high:      0.75,
    ConfidenceLevel.medium:    0.50,
    ConfidenceLevel.low:       0.25,
}


def calibrate_extraction(extraction: MedicalExtraction) -> dict:
    """
    Takes the full MedicalExtraction object and returns the familiar flat
    JSON format, with each finding now enhanced with a calibrated_score float.
    """
    calibrated_findings = []
    for f in extraction.findings:
        score = CONFIDENCE_MAP.get(f.confidence_level, 0.50)
        calibrated_findings.append({
            "finding":            f.finding,
            "confidence_level":   f.confidence_level.value,
            "calibrated_score":   score,
            "supporting_evidence": f.supporting_evidence,
        })

    # Sort by calibrated_score descending — most confident finding first
    calibrated_findings.sort(key=lambda x: x["calibrated_score"], reverse=True)

    return {
        "modality":             extraction.modality,
        "anatomy":              extraction.anatomy,
        "image_quality": {
            "usable":             extraction.image_quality.usable,
            "issues": {
                "blur":         extraction.image_quality.issues.blur,
                "low_contrast": extraction.image_quality.issues.low_contrast,
                "glare":        extraction.image_quality.issues.glare,
                "cropped":      extraction.image_quality.issues.cropped,
            },
            "reason_if_unusable": extraction.image_quality.reason_if_unusable,
        },
        "uncertainty_sources": {
            "image_quality_issues":     extraction.uncertainty_sources.image_quality_issues,
            "visual_evidence_strength": f"{extraction.uncertainty_sources.visual_evidence_strength}%",
            "model_ambiguity":          extraction.uncertainty_sources.model_ambiguity,
            "out_of_distribution": {
                "detected": extraction.uncertainty_sources.out_of_distribution.detected,
                "reason":   extraction.uncertainty_sources.out_of_distribution.reason,
            },
        },
        "findings":             calibrated_findings,
        "retrieval_queries":    extraction.retrieval_queries,
        "recommended_pipeline": extraction.recommended_pipeline,
    }
