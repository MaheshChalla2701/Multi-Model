import os
import json
import logging
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class RecommendedStep(BaseModel):
    action: str = Field(description="The recommended clinical action (e.g., 'Order contrast MRI', 'Immediate biopsy').")
    urgency: str = Field(description="Urgency level: 'Routine', 'Urgent', or 'Emergent'.")

class FinalReport(BaseModel):
    primary_diagnosis: str = Field(description="The most likely diagnosis based on all AI models and historical evidence.")
    confidence_score: float = Field(description="A confidence percentage between 0.0 and 1.0.")
    differential_diagnoses: list[str] = Field(description="A list of 2-3 other possible diagnoses.")
    reasoning: str = Field(description="A detailed explanation of why the primary diagnosis was chosen, citing specific findings from the specialist and historical/guideline evidence.")
    recommended_steps: list[RecommendedStep] = Field(description="List of recommended next clinical steps.")

def _get_genai_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_google_ai_studio_key_here":
        return None
    return genai.Client(api_key=api_key)

async def synthesize_final_report(
    generalist_data: dict, 
    specialist_data: dict, 
    historical_cases: list, 
    graph_guidelines: list
) -> dict:
    """
    Acts as the Chief Medical Officer (CMO).
    Synthesizes the generalist analysis, specialist deep-dive, vector historical cases,
    and graph medical guidelines into one cohesive, highly confident final report.
    """
    logger.info("Synthesizing final diagnostic report...")
    
    client = _get_genai_client()
    if not client:
        logger.warning("No Gemini API key found for Inference Engine. Using mock report.")
        return _mock_final_report()
        
    prompt = f"""
    You are an expert Chief Medical Officer AI. Your job is to review the following data from a multi-agent diagnostic pipeline and synthesize it into a final, highly accurate clinical report.

    1. GENERALIST ANALYSIS (Broad anatomical overview):
    {json.dumps(generalist_data, indent=2)}

    2. SPECIALIST ANALYSIS (Deep dive into pathology):
    {json.dumps(specialist_data, indent=2)}

    3. HISTORICAL EVIDENCE (Top 5 similar past cases from Vector DB):
    {json.dumps(historical_cases, indent=2)}

    4. CLINICAL GUIDELINES (Medical rules from Graph DB):
    {json.dumps(graph_guidelines, indent=2)}

    Based STRICTLY on this data, provide your final diagnostic conclusion. Do not hallucinate outside the provided evidence. 
    Ensure the confidence score reflects the consistency between the specialist finding, historical evidence, and guidelines.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalReport,
                temperature=0.1, # Keep it highly deterministic for medical safety
            ),
        )
        # Parse the JSON response
        result = json.loads(response.text)
        return result
    except Exception as e:
        logger.error(f"Inference Engine failed: {e}")
        return _mock_final_report()

def _mock_final_report() -> dict:
    return {
        "primary_diagnosis": "Indeterminate Lesion",
        "confidence_score": 0.5,
        "differential_diagnoses": ["Artifact", "Benign mass"],
        "reasoning": "Mock fallback used because Gemini API key was missing or inference failed.",
        "recommended_steps": [
            {"action": "Clinical Correlation", "urgency": "Routine"}
        ]
    }
