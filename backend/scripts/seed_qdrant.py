import os
import uuid
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from google import genai

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MOCK_CASES = [
    {
        "clinical_notes": "55yo male with severe left-sided chest pain radiating to back. Elevated d-dimer.",
        "diagnosis": "Acute Pulmonary Embolism",
        "findings": "Filling defect in left main pulmonary artery extending into lower lobe branches."
    },
    {
        "clinical_notes": "42yo female with chronic headaches. No focal neurological deficits.",
        "diagnosis": "Meningioma",
        "findings": "Well-circumscribed extra-axial mass with homogeneous enhancement along the falx cerebri."
    },
    {
        "clinical_notes": "65yo male, trauma. Fell from ladder.",
        "diagnosis": "Colles Fracture",
        "findings": "Dorsally displaced extra-articular fracture of the distal radius."
    },
    {
        "clinical_notes": "50yo female, screening mammogram. No symptoms.",
        "diagnosis": "Invasive Ductal Carcinoma",
        "findings": "Spiculated mass in the upper outer quadrant of the right breast with associated pleomorphic microcalcifications."
    }
]

def main():
    if not QDRANT_URL or not QDRANT_API_KEY:
        logger.error("Missing Qdrant Cloud credentials in .env. Cannot seed.")
        return
        
    if not GEMINI_API_KEY:
        logger.error("Missing GEMINI_API_KEY in .env. Cannot generate embeddings.")
        return

    logger.info("Initializing clients...")
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    gemini = genai.Client(api_key=GEMINI_API_KEY)

    collection_name = "medical_cases"

    # Recreate collection (destructive for demo purposes)
    try:
        qdrant.delete_collection(collection_name)
    except Exception:
        pass
        
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    logger.info(f"Created collection '{collection_name}'")

    points = []
    for case in MOCK_CASES:
        logger.info(f"Embedding case: {case['diagnosis']}...")
        # We embed the findings and clinical notes so they match when searching for evidence
        text_to_embed = f"{case['clinical_notes']} {case['findings']}"
        response = gemini.models.embed_content(
            model='gemini-embedding-2',
            contents=text_to_embed
        )
        
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=response.embeddings[0].values,
                payload={
                    "case_id": f"CASE-{str(uuid.uuid4())[:8].upper()}",
                    "diagnosis": case["diagnosis"],
                    "clinical_notes": case["clinical_notes"],
                    "findings": case["findings"]
                }
            )
        )

    logger.info(f"Upserting {len(points)} points to Qdrant...")
    qdrant.upsert(collection_name=collection_name, points=points)
    logger.info("Seeding complete!")

if __name__ == "__main__":
    main()
