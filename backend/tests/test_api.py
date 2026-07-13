"""
test_api.py
-----------
Integration tests for the FastAPI endpoints.
Uses httpx.AsyncClient so no real server process is needed.
External services (Gemini, Qdrant, Neo4j) are patched out.
"""

import io
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport


# ── Import the app AFTER conftest has seeded env vars ────────────────────────
from app.main import app


# ── Minimal mock data ─────────────────────────────────────────────────────────
_MOCK_GEMINI_RESULT = {
    "modality": "Xray",
    "anatomy": "Chest",
    "uncertainty_sources": {
        "image_quality_issues": [],
        "visual_evidence_strength": "80%",
        "model_ambiguity": False,
        "out_of_distribution": {"detected": False, "reason": None},
    },
    "findings": [],
    "retrieval_queries": [],
    "recommended_pipeline": "Chest_Xray",
}

_MOCK_PIPELINE_OUTPUT = MagicMock()
_MOCK_PIPELINE_OUTPUT.model_dump.return_value = {"pipeline_name": "Chest X-Ray Pipeline", "key_findings": []}
_MOCK_PIPELINE_OUTPUT.retrieval_queries = []

_MOCK_FINAL_REPORT = {
    "primary_diagnosis": "Normal study",
    "confidence_score": 0.9,
    "differential_diagnoses": [],
    "reasoning": "No abnormalities detected.",
    "recommended_steps": [],
    "safety_disclaimer": "AI disclaimer text.",
    "critical_alerts": [],
    "safety_verified": True,
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── Health check ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_check(client):
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data


# ── /api/analyze-scan: validation errors ─────────────────────────────────────

@pytest.mark.asyncio
async def test_analyze_scan_rejects_non_image(client):
    """Non-image, non-DICOM files must be rejected with 400."""
    response = await client.post(
        "/api/analyze-scan",
        files={"file": ("report.pdf", b"%PDF-fake", "application/pdf")},
        data={"age": "", "sex": "", "symptoms": ""},
    )
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower() or "dicom" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_analyze_scan_rejects_oversized_file(client):
    """Files larger than 50 MB must be rejected with 413."""
    big_payload = b"\x00" * (51 * 1024 * 1024)
    response = await client.post(
        "/api/analyze-scan",
        files={"file": ("scan.png", big_payload, "image/png")},
        data={"age": "", "sex": "", "symptoms": ""},
    )
    assert response.status_code == 413


# ── /api/analyze-scan: happy path (all external calls patched) ────────────────

@pytest.mark.asyncio
@patch("app.main.analyze_medical_image", return_value=dict(_MOCK_GEMINI_RESULT))
@patch("app.main.domain_router.route", return_value=_MOCK_PIPELINE_OUTPUT)
@patch("app.main.search_evidence", new_callable=AsyncMock, return_value=[])
@patch("app.main.query_knowledge_graph", new_callable=AsyncMock, return_value=[])
@patch("app.main.synthesize_final_report", new_callable=AsyncMock, return_value=dict(_MOCK_FINAL_REPORT))
@patch("app.main.log_case", new_callable=AsyncMock)
async def test_analyze_scan_success(
    mock_log, mock_synthesize, mock_neo4j, mock_qdrant, mock_router, mock_gemini, client
):
    """A valid image upload should return 200 with the expected top-level keys."""
    # Minimal 1×1 white PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    response = await client.post(
        "/api/analyze-scan",
        files={"file": ("scan.png", png_bytes, "image/png")},
        data={"age": "45", "sex": "Male", "symptoms": "cough"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "modality" in data
    assert "anatomy" in data
    assert "findings" in data
    assert "final_diagnosis_report" in data
    assert "case_id" in data


# ── /api/grid-preview ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_grid_preview_rejects_non_image(client):
    response = await client.post(
        "/api/grid-preview",
        files={"file": ("doc.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_grid_preview_rejects_oversized_file(client):
    big_payload = b"\x00" * (51 * 1024 * 1024)
    response = await client.post(
        "/api/grid-preview",
        files={"file": ("scan.png", big_payload, "image/png")},
    )
    assert response.status_code == 413
