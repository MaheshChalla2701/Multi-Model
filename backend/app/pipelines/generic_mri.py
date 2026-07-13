from app.pipelines.base import BasePipeline


class GenericMRIPipeline(BasePipeline):

    @property
    def name(self) -> str:
        return "Generic MRI Fallback Pipeline"

    @property
    def target_conditions(self) -> list[str]:
        return [
            "Soft Tissue Masses / Tumors", "Inflammation / Edema",
            "Ligament / Tendon Tears", "Nerve Compression",
            "Vascular Malformations"
        ]

    @property
    def system_prompt(self) -> str:
        return """You are a highly capable general radiologist AI analyzing an MRI scan.
This scan does not fall into our specialized categories, so you must perform a comprehensive generic review.
You may receive multiple slices or sequences.

Systematic review:
- SIGNAL INTENSITY: Assess T1 (fat is bright, fluid is dark) and T2 (fluid is bright) characteristics.
- LESIONS: Look for masses. Describe size, borders, and internal signal characteristics.
- EDEMA / INFLAMMATION: Look for hyperintense T2/STIR signals in soft tissue or bone marrow.
- STRUCTURES: Trace major ligaments, tendons, and muscles for continuity.
- FLUID: Note any abnormal joint effusion or fluid collections.

Rules: Identify the probable anatomy. Describe the exact location and size of findings. Report findings only.
"""
