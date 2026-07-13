import os
import logging
from typing import Any
from qdrant_client import QdrantClient
from google import genai

logger = logging.getLogger(__name__)

# Initialize clients lazily or handle missing keys gracefully
_qdrant_client = None
_genai_client = None


def _get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        url = os.getenv("QDRANT_URL")
        api_key = os.getenv("QDRANT_API_KEY")
        if url and api_key:
            try:
                _qdrant_client = QdrantClient(url=url, api_key=api_key)
                logger.info("Qdrant Cloud client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize QdrantClient: {e}")
                return None
    return _qdrant_client


def _get_genai_client():
    global _genai_client
    if _genai_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            _genai_client = genai.Client(api_key=api_key)
    return _genai_client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed text queries using Gemini."""
    client = _get_genai_client()
    if not client:
        logger.warning("No Gemini API key found for embeddings. Using mock embeddings.")
        return [[0.0] * 3072 for _ in texts]
    
    embeddings = []
    for text in texts:
        try:
            # text-embedding-004 returns a 768-dimensional vector
            response = client.models.embed_content(
                model='gemini-embedding-2',
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        except Exception as e:
            logger.error(f"Embedding failed for '{text}': {e}")
            embeddings.append([0.0] * 3072)
            
    return embeddings


async def search_evidence(queries: list[str], limit: int = 3) -> list[dict[str, Any]]:
    """
    Search Qdrant for similar historical medical cases or literature.
    Returns a consolidated list of evidence documents.
    """
    if not queries:
        return []

    client = _get_qdrant_client()
    
    # MOCK MODE: If the user hasn't set up Qdrant yet, return fake evidence
    if not client:
        logger.warning("QDRANT_URL/API_KEY missing. Returning mock evidence.")
        return [
            {
                "case_id": "MOCK-CASE-992",
                "similarity": 0.92,
                "diagnosis": "Mock Diagnosis based on findings",
                "matched_finding": q,
                "notes": "This is a mock record because Qdrant is not configured."
            } for q in queries[:2]
        ]

    # REAL MODE: Generate embeddings and search Qdrant
    evidence_results = []
    query_embeddings = get_embeddings(queries)
    
    for i, embedding in enumerate(query_embeddings):
        # Skip if embedding failed (all zeros)
        if all(v == 0.0 for v in embedding):
            continue
            
        try:
            search_result = client.search(
                collection_name="medical_cases",
                query_vector=embedding,
                limit=limit
            )
            for hit in search_result:
                evidence_results.append({
                    "case_id": hit.payload.get("case_id", "Unknown"),
                    "similarity": round(hit.score, 4),
                    "diagnosis": hit.payload.get("diagnosis", "Unknown"),
                    "matched_finding": queries[i],
                    "notes": hit.payload.get("clinical_notes", "")
                })
        except Exception as e:
            logger.error(f"Qdrant search failed for query '{queries[i]}': {e}")

    # Sort consolidated results by highest similarity score
    evidence_results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top 5 unique pieces of evidence across all queries
    return evidence_results[:5]
