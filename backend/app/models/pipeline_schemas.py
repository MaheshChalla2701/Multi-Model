from pydantic import BaseModel, Field
from typing import List, Optional


class AbnormalRegion(BaseModel):
    """
    Describes a specific abnormal region found in a frame.
    Used as a placeholder for future MedSAM3 segmentation masks.
    """
    frame_index: Optional[int] = Field(
        None,
        description="Index of the frame (0-based) in which this region was found. None for single-frame images."
    )
    location: str = Field(
        description="Anatomical location of the abnormality. E.g., 'right lower lobe', 'left temporal region'."
    )
    description: str = Field(
        description="Detailed visual description of what was observed in this region."
    )
    severity: str = Field(
        description="Estimated severity: 'mild', 'moderate', or 'severe'."
    )


class PipelineOutput(BaseModel):
    """
    Structured output from a specialist pipeline after domain-specific analysis.
    Passed downstream to the Evidence Collection and Inference layers.
    """
    pipeline_name: str = Field(
        description="Name of the specialist pipeline that processed this study."
    )
    analyzed_frames: int = Field(
        description="Total number of frames that were analyzed."
    )
    abnormal_regions: List[AbnormalRegion] = Field(
        description="List of detected abnormal regions across all frames. Empty if normal."
    )
    key_findings: List[str] = Field(
        description=(
            "Domain-specific findings identified by the specialist pipeline. "
            "E.g., ['Cortical disruption at distal radius', 'Pneumothorax right side']. "
            "More specific than the Layer 3 findings."
        )
    )
    retrieval_queries: List[str] = Field(
        description=(
            "Refined retrieval queries for Qdrant, generated from the specialist analysis. "
            "More specific than the Layer 3 queries. E.g., ['distal radius fracture xray', 'right pneumothorax CT']."
        )
    )
    normal_study: bool = Field(
        description="True if no abnormalities were found by this specialist pipeline."
    )
    specialist_notes: Optional[str] = Field(
        None,
        description="Any additional clinical notes or observations from the specialist pipeline."
    )
