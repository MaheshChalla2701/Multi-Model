from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import json
from app.models.schemas import MedicalExtraction
from app.services.calibration import calibrate_extraction

load_dotenv(override=True)

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not set in .env")

client = genai.Client(api_key=api_key)

SYSTEM_PROMPT = """
You are an expert medical imaging analysis system with deep radiology knowledge.
Your role is to transform a raw medical image into structured, evidence-based observations.

## Your 5-step workflow:

### Step 1 – Image Quality Assessment
Evaluate the image for:
- Blur or motion artifacts
- Low resolution or low contrast
- Partial cropping or rotation
- Whether it is a phone photo of a scan (out-of-distribution)
- Whether it is actually a medical image at all

### Step 2 – Modality & Anatomy Classification
Identify: X-ray / MRI / CT / Ultrasound and the primary anatomical region.

### Step 3 – Uncertainty Source Identification
Before listing findings, flag which uncertainty sources are active:
- image_quality_issues: Is the image quality degraded?
- weak_visual_evidence: Are findings subtle or borderline?
- model_ambiguity: Can the same pattern suggest multiple different findings?
- out_of_distribution: Is this an atypical scan (unusual angle, phone photo, rare modality)?

### Step 4 – Structured Finding Extraction
For EACH distinct abnormal finding you observe:
- Name the finding precisely (e.g., "Pleural Effusion", not "fluid")
- Assign a confidence_level: very_high / high / medium / low
  - very_high: unmistakably clear, textbook example
  - high: clearly visible, minimal doubt
  - medium: probable but subtle, could be normal variant
  - low: possible but speculative, needs clinical correlation
- List the specific visual signs (supporting_evidence) that led you to this finding
  (e.g., "blunted costophrenic angle", "meniscus sign", "basal opacity")
  These must be pixel-level observations, NOT the finding name itself.
- Generate retrieval_queries for a vector database

### Step 5 – Pipeline Routing
Based on modality + anatomy, recommend the appropriate specialist pipeline.

## Rules:
- Report findings ONLY — no diagnoses, no treatment recommendations.
- If the image is normal with no abnormal findings, set normal_study: true and findings: []
- If the image is unusable, set usable: false and findings: []
- NEVER invent findings that are not clearly supported by visual evidence in the image.
"""


def analyze_medical_image(image_bytes: bytes, mime_type: str, patient_context: str) -> dict:
    try:
        prompt = (
            f"Patient Context: {patient_context if patient_context else 'None provided.'}\n\n"
            "Analyze the provided medical image following your 5-step workflow. "
            "Be precise. Only report what you can visually verify."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=MedicalExtraction,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
            ),
        )

        # Parse the structured Pydantic object
        if hasattr(response, 'parsed') and response.parsed:
            extraction: MedicalExtraction = response.parsed
        else:
            # Fallback: parse from raw JSON
            raw = json.loads(response.text)
            extraction = MedicalExtraction(**raw)

        # Run through our calibration layer to produce float scores
        return calibrate_extraction(extraction)

    except Exception as e:
        raise Exception(f"Failed to analyze image with Gemini: {str(e)}")
