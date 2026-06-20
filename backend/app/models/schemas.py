from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class ConfidenceLevel(str, Enum):
    very_high = "very_high"
    high      = "high"
    medium    = "medium"
    low       = "low"


# ── Image Quality ─────────────────────────────────────────────────────────────

class ImageQualityIssues(BaseModel):
    blur: bool = Field(
        description="True if the image shows blur or motion artifacts that reduce sharpness."
    )
    low_contrast: bool = Field(
        description="True if the image has poor contrast between structures, making features hard to distinguish."
    )
    glare: bool = Field(
        description="True if the image has overexposed regions, reflections, or glare obscuring detail."
    )
    cropped: bool = Field(
        description="True if relevant anatomy appears partially cut off or outside the image frame."
    )


class ImageQuality(BaseModel):
    usable: bool = Field(
        description="True if the image is of sufficient quality for analysis, False otherwise."
    )
    issues: ImageQualityIssues = Field(
        description=(
            "Specific quality degradation flags. Set each to true only if that issue is "
            "clearly present and affects analysis. All false if the image is clean."
        )
    )
    reason_if_unusable: Optional[str] = Field(
        None,
        description="If usable is False, explain why (e.g., 'Too blurry', 'Not a medical image')."
    )


# ── Uncertainty Sources ───────────────────────────────────────────────────────

class OutOfDistribution(BaseModel):
    detected: bool = Field(
        description=(
            "True if the image is atypical compared to standard training examples: "
            "e.g., a phone photo of a printed scan, unusual projection angle, "
            "rare modality, or heavily post-processed image."
        )
    )
    reason: Optional[str] = Field(
        None,
        description=(
            "If detected is True, describe why the image is out-of-distribution. "
            "E.g., 'Phone photo of a printed X-ray with visible paper texture and glare', "
            "'Oblique projection instead of standard PA view', 'Heavy JPEG compression artifacts'."
        )
    )


class UncertaintySources(BaseModel):
    image_quality_issues: bool = Field(
        description=(
            "True if any image quality degradation (blur, low contrast, glare, cropping) "
            "is present that may impact finding detection accuracy."
        )
    )
    visual_evidence_strength: int = Field(
        description=(
            "Integer 0–100 representing how clearly the findings are visible in the image. "
            "100 = textbook-clear, unmistakable visual evidence. "
            "75 = clearly visible with minor uncertainty. "
            "50 = present but subtle, borderline. "
            "25 = faint, speculative, barely discernible. "
            "0 = no visual evidence at all. "
            "Base this on the weakest finding if multiple are present."
        )
    )
    model_ambiguity: bool = Field(
        description=(
            "True if the same visual pattern could plausibly support two or more "
            "different radiological interpretations."
        )
    )
    out_of_distribution: OutOfDistribution = Field(
        description="Whether the image differs significantly from typical training examples."
    )


# ── Findings ──────────────────────────────────────────────────────────────────

class StructuredFinding(BaseModel):
    finding: str = Field(
        description="A single specific radiological observation. E.g., 'Pleural Effusion'. Not a diagnosis."
    )
    confidence_level: ConfidenceLevel = Field(
        description=(
            "Ordinal confidence that this finding is present: "
            "very_high (unmistakably clear), high (clearly visible), "
            "medium (probable but subtle), low (possible but uncertain)."
        )
    )
    supporting_evidence: List[str] = Field(
        description=(
            "Specific low-level visual signs from the image that support this finding. "
            "E.g., ['blunted costophrenic angle', 'meniscus sign', 'basal opacity']. "
            "These are pixel-level observations, NOT the finding name repeated."
        )
    )


# ── Top-Level Extraction ──────────────────────────────────────────────────────

class MedicalExtraction(BaseModel):
    modality: str = Field(
        description="The imaging modality. One of: 'Xray', 'MRI', 'CT', 'Ultrasound', 'Unknown'."
    )
    anatomy: str = Field(
        description="The anatomical region imaged. E.g., 'Chest', 'Brain', 'Knee', 'Spine', 'Abdomen'."
    )
    image_quality: ImageQuality = Field(
        description="Assessment of the image quality, including specific degradation flags."
    )
    uncertainty_sources: UncertaintySources = Field(
        description=(
            "Quantified and qualified sources of uncertainty active for this image. "
            "Be precise — these directly affect downstream confidence in findings."
        )
    )
    findings: List[StructuredFinding] = Field(
        description=(
            "List of structured findings. Each finding includes a confidence level "
            "and the specific visual evidence supporting it. "
            "Empty list if no abnormal findings are visible."
        )
    )
    retrieval_queries: List[str] = Field(
        description=(
            "Flat list of search queries for a vector database, generated from all findings. "
            "E.g., ['bilateral pleural effusion chest xray', 'blunted costophrenic angle']."
        )
    )
    recommended_pipeline: str = Field(
        description=(
            "Specialist pipeline to route to. "
            "E.g., 'Chest_Xray', 'Neuro_MRI', 'Spine_MRI', 'Bone_Xray', 'Dental_Xray', "
            "'Neuro_CT', 'Chest_CT', 'Abdomen_CT', 'Trauma_CT', 'Abdominal_MRI', 'MSK_MRI'."
        )
    )
