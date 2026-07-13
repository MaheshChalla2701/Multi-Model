"""
Base Specialist Pipeline (Layer 5)
-----------------------------------
All specialist pipelines inherit from BasePipeline.

They share the same Gemini Flash-powered analysis engine but each subclass
provides a domain-specific system prompt and target conditions list.

When MedSAM3 is integrated later, only _segment() needs to be overridden.
"""

import json
import os
from abc import ABC, abstractmethod

from google import genai
from google.genai import types
from dotenv import load_dotenv

from app.models.pipeline_schemas import AbnormalRegion, PipelineOutput

load_dotenv(override=True)

# Lazy-initialised so the client is never built at import time
# (before dotenv has had a chance to populate GEMINI_API_KEY).
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment.")
        _client = genai.Client(api_key=api_key)
    return _client

# Shared response schema for all pipelines
_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "normal_study": {
            "type": "boolean",
            "description": "True if no abnormalities are found."
        },
        "key_findings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of specific domain findings identified."
        },
        "abnormal_regions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame_index":  {"type": "integer"},
                    "location":     {"type": "string"},
                    "description":  {"type": "string"},
                    "severity":     {"type": "string", "enum": ["mild", "moderate", "severe"]}
                },
                "required": ["location", "description", "severity"]
            }
        },
        "retrieval_queries": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific Qdrant retrieval queries for this domain."
        },
        "specialist_notes": {
            "type": "string",
            "description": "Any additional clinical observations."
        }
    },
    "required": ["normal_study", "key_findings", "abnormal_regions", "retrieval_queries"]
}


class BasePipeline(ABC):
    """
    Abstract base class for all specialist pipelines.
    Subclasses must define `name`, `system_prompt`, and `target_conditions`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable pipeline name, e.g., 'Chest X-Ray Pipeline'."""
        ...

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Domain-specific system prompt for Gemini Flash."""
        ...

    @property
    @abstractmethod
    def target_conditions(self) -> list[str]:
        """List of conditions this pipeline specifically screens for."""
        ...

    def run(
        self,
        frames: bytes | list[bytes],
        mime_type: str,
        layer3_findings: list[str],
        patient_context: str,
    ) -> PipelineOutput:
        """
        Main entry point. Analyzes frames with domain-specific Gemini prompt.
        Returns structured PipelineOutput.
        """
        images = frames if isinstance(frames, list) else [frames]
        num_frames = len(images)

        # Build the user prompt incorporating Layer 3 findings for continuity
        findings_text = "\n".join(f"- {f}" for f in layer3_findings) if layer3_findings else "None reported."
        conditions_text = ", ".join(self.target_conditions)

        user_prompt = (
            f"Patient Context:\n{patient_context}\n\n"
            f"Layer 3 Pre-screening Findings:\n{findings_text}\n\n"
            f"You are the {self.name}. Perform a detailed domain-specific analysis of the provided image(s).\n"
            f"Specifically screen for these conditions: {conditions_text}.\n\n"
            f"If multiple images are provided, they are sequential frames from the same study. "
            f"For each abnormal region found, specify the frame_index (0-based).\n"
            f"Report only what is clearly visible. Do not invent findings."
        )

        # Build multimodal content
        contents = []
        for img_bytes in images:
            contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        contents.append(user_prompt)

        try:
            response = _get_client().models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.system_prompt,
                    response_mime_type="application/json",
                    response_schema=_PIPELINE_SCHEMA,
                    temperature=0.1,
                ),
            )

            raw = json.loads(response.text)

            abnormal_regions = [
                AbnormalRegion(
                    frame_index=r.get("frame_index"),
                    location=r["location"],
                    description=r["description"],
                    severity=r["severity"],
                )
                for r in raw.get("abnormal_regions", [])
            ]

            return PipelineOutput(
                pipeline_name=self.name,
                analyzed_frames=num_frames,
                abnormal_regions=abnormal_regions,
                key_findings=raw.get("key_findings", []),
                retrieval_queries=raw.get("retrieval_queries", []),
                normal_study=raw.get("normal_study", len(abnormal_regions) == 0),
                specialist_notes=raw.get("specialist_notes"),
            )

        except Exception as e:
            # On failure, pass Layer 3 findings through so the pipeline never blocks
            return PipelineOutput(
                pipeline_name=self.name,
                analyzed_frames=num_frames,
                abnormal_regions=[],
                key_findings=layer3_findings,
                retrieval_queries=layer3_findings,
                normal_study=False,
                specialist_notes=f"Pipeline analysis error: {str(e)}. Layer 3 findings used as fallback.",
            )
