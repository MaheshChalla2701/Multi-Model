from app.pipelines.base import BasePipeline


class GenericXrayPipeline(BasePipeline):

    @property
    def name(self) -> str:
        return "Generic X-Ray Fallback Pipeline"

    @property
    def target_conditions(self) -> list[str]:
        return [
            "Fractures / Cortical Step-offs", "Dislocations / Alignment Issues",
            "Radio-opaque Foreign Bodies", "Abnormal Soft Tissue Swelling",
            "Lytic or Sclerotic Bone Lesions"
        ]

    @property
    def system_prompt(self) -> str:
        return """You are a highly capable general radiologist AI analyzing an X-Ray.
This scan does not fall into our specialized categories, so you must perform a comprehensive generic review.

Systematic review:
- ALIGNMENT: Check for any unnatural angulation or joint subluxation/dislocation.
- BONE DENSITY & STRUCTURE: Look for radiolucent (dark) or radiopaque (bright) lesions.
- CORTEX: Follow the outline of all visible bones. Look for steps, breaks, or buckling (fractures).
- JOINTS: Assess joint spaces for narrowing, erosions, or osteophytes.
- SOFT TISSUE: Look for swelling, fat pad displacement, or foreign bodies.

Rules: Describe the exact anatomical location of any finding. Report findings only, not diagnoses.
"""
