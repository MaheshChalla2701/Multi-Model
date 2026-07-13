"""
Domain Router (Layer 4)
-----------------------
Reads the `recommended_pipeline` field from the Gemini extraction output
and routes the study to the correct specialist pipeline class.

All pipelines follow the same contract:
    run(frames: bytes | list[bytes], mime_type: str, layer3_findings: list, patient_context: str) -> PipelineOutput
"""

from app.models.pipeline_schemas import PipelineOutput


# Registry maps pipeline name strings → pipeline module import paths
# Populated lazily on first call to avoid circular imports
_PIPELINE_REGISTRY: dict[str, type] = {}


def _build_registry():
    """Lazy-load all pipeline classes into the registry."""
    global _PIPELINE_REGISTRY
    if _PIPELINE_REGISTRY:
        return

    from app.pipelines.chest_xray      import ChestXrayPipeline
    from app.pipelines.chest_ct        import ChestCTPipeline
    from app.pipelines.neuro_mri       import NeuroMRIPipeline
    from app.pipelines.neuro_ct        import NeuroCTPipeline
    from app.pipelines.bone_xray       import BoneXrayPipeline
    from app.pipelines.spine_mri       import SpineMRIPipeline
    from app.pipelines.msk_mri         import MSKMRIPipeline
    from app.pipelines.abdominal_mri   import AbdominalMRIPipeline
    from app.pipelines.abdomen_ct      import AbdomenCTPipeline
    from app.pipelines.trauma_ct       import TraumaCTPipeline
    from app.pipelines.dental_xray     import DentalXrayPipeline
    
    # Newly added specialists
    from app.pipelines.mammography       import MammographyPipeline
    from app.pipelines.cardiac_mri       import CardiacMRIPipeline
    from app.pipelines.fetal_ultrasound  import FetalUltrasoundPipeline

    # Generic Fallbacks
    from app.pipelines.generic_xray       import GenericXrayPipeline
    from app.pipelines.generic_ct         import GenericCTPipeline
    from app.pipelines.generic_mri        import GenericMRIPipeline
    from app.pipelines.generic_ultrasound import GenericUltrasoundPipeline

    _PIPELINE_REGISTRY = {
        # X-Ray & Mammography
        "Chest_Xray":       ChestXrayPipeline,
        "Bone_Xray":        BoneXrayPipeline,
        "Dental_Xray":      DentalXrayPipeline,
        "Mammography":      MammographyPipeline,

        # MRI pipelines
        "Neuro_MRI":        NeuroMRIPipeline,
        "Spine_MRI":        SpineMRIPipeline,
        "MSK_MRI":          MSKMRIPipeline,
        "Abdominal_MRI":    AbdominalMRIPipeline,
        "Cardiac_MRI":      CardiacMRIPipeline,

        # CT pipelines
        "Neuro_CT":         NeuroCTPipeline,
        "Chest_CT":         ChestCTPipeline,
        "Abdomen_CT":       AbdomenCTPipeline,
        "Trauma_CT":        TraumaCTPipeline,

        # Ultrasound
        "Fetal_Ultrasound": FetalUltrasoundPipeline,

        # Generic Fallbacks (internal use)
        "_GENERIC_XRAY":       GenericXrayPipeline,
        "_GENERIC_CT":         GenericCTPipeline,
        "_GENERIC_MRI":        GenericMRIPipeline,
        "_GENERIC_ULTRASOUND": GenericUltrasoundPipeline,
    }


def route(
    recommended_pipeline: str,
    frames: bytes | list[bytes],
    mime_type: str,
    layer3_findings: list[str],
    patient_context: str,
) -> PipelineOutput:
    """
    Routes the study to the correct specialist pipeline and returns its output.

    Args:
        recommended_pipeline: The pipeline name from Gemini's Layer 3 output.
        frames: Raw image bytes or a list of frame bytes (for multi-slice).
        mime_type: MIME type of the frames (e.g., 'image/png').
        layer3_findings: Findings already extracted by Gemini in Layer 3.
        patient_context: Structured patient context string.

    Returns:
        PipelineOutput with domain-specific findings and abnormal regions.
    """
    _build_registry()

    pipeline_class = _PIPELINE_REGISTRY.get(recommended_pipeline)

    if pipeline_class is None:
        name_upper = recommended_pipeline.upper()
        if "XRAY" in name_upper or "X_RAY" in name_upper or "X-RAY" in name_upper:
            pipeline_class = _PIPELINE_REGISTRY.get("_GENERIC_XRAY")
        elif "CT" in name_upper:
            pipeline_class = _PIPELINE_REGISTRY.get("_GENERIC_CT")
        elif "MRI" in name_upper:
            pipeline_class = _PIPELINE_REGISTRY.get("_GENERIC_MRI")
        elif "ULTRASOUND" in name_upper or "US" in name_upper:
            pipeline_class = _PIPELINE_REGISTRY.get("_GENERIC_ULTRASOUND")

    if pipeline_class is None:
        # Ultimate Fallback: completely unknown modality — return a minimal pass-through output
        return PipelineOutput(
            pipeline_name=f"Unknown ({recommended_pipeline})",
            analyzed_frames=len(frames) if isinstance(frames, list) else 1,
            abnormal_regions=[],
            key_findings=layer3_findings,
            retrieval_queries=layer3_findings,
            normal_study=len(layer3_findings) == 0,
            specialist_notes=(
                f"No specialist or generic fallback found for '{recommended_pipeline}'. "
                "Layer 3 findings passed through directly."
            ),
        )

    pipeline = pipeline_class()
    return pipeline.run(frames, mime_type, layer3_findings, patient_context)
