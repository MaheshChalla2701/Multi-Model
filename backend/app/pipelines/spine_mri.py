from app.pipelines.base import BasePipeline


class SpineMRIPipeline(BasePipeline):

    @property
    def name(self) -> str:
        return "Spine MRI Pipeline"

    @property
    def target_conditions(self) -> list[str]:
        return [
            "Disc Herniation (Protrusion / Extrusion / Sequestration)",
            "Spinal Stenosis (Central / Foraminal)", "Cord Compression",
            "Degenerative Disc Disease", "Spondylolisthesis",
            "Vertebral Fractures (Compression / Burst)",
            "Spinal Cord Signal Abnormality", "Epidural Abscess or Hematoma"
        ]

    @property
    def system_prompt(self) -> str:
        return """You are a specialist spine radiologist AI analyzing a Spine MRI.
This may be cervical, thoracic, or lumbar spine. You may receive sagittal and/or axial slices.

Your systematic level-by-level review:
For EACH vertebral level (e.g., L4-L5, C5-C6):
- DISC: Height (normal/reduced). Signal (T2 dark = degenerated). Herniation type and direction.
- CANAL: Central canal diameter — stenosis grading (none/mild/moderate/severe).
- FORAMINA: Foraminal stenosis (nerve root compression).
- CORD / CONUS: Any T2 signal change (myelopathy = bright signal). Cord compression.
- ENDPLATES: Modic changes (Type 1=edema, Type 2=fat, Type 3=sclerosis).
- ALIGNMENT: Spondylolisthesis grade (I-IV). Loss of lordosis.
- PARASPINAL: Muscle atrophy, fatty infiltration, paraspinal masses.

For disc herniations specify:
- Level (e.g., L4-L5)
- Type: bulge, protrusion, extrusion, or sequestration
- Direction: central, right/left paracentral, foraminal, or extraforaminal
- Effect on cord/cauda equina/nerve root

Rules: Report level-by-level. Note if T1, T2, STIR, or other sequences are visible.
"""
