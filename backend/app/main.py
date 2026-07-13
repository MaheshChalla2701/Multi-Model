import asyncio
import os
import uuid

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app.services.gemini_service import analyze_medical_image
from app.services import router as domain_router
from app.services.qdrant_service import search_evidence
from app.services.neo4j_service import query_knowledge_graph
from app.services.inference_engine import synthesize_final_report
from app.services.safety_service import verify_report_safety
from app.services.audit_service import log_case
from app.utils.dicom_utils import read_dicom_bytes
from app.utils.slicing import assess_objective_quality, count_grid_slices, draw_grid_on_image

# ── CORS ─────────────────────────────────────────────────────────────────────
# Read allowed origins from env so this works in dev, staging, and production
# without code changes. Comma-separate multiple origins in the env var.
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app = FastAPI(title="Medical Understanding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Maximum accepted file size (50 MB) enforced server-side
MAX_FILE_BYTES = 50 * 1024 * 1024


@app.get("/")
def health_check():
    return {"status": "ok", "service": "Medical Understanding API"}


@app.post("/api/analyze-scan")
async def analyze_scan(
    file: UploadFile = File(...),
    age: str = Form(""),
    sex: str = Form(""),
    symptoms: str = Form(""),
):
    """
    Primary analysis endpoint (Layers 1 – 10).

    Accepts a medical image or DICOM file plus structured patient context.
    Extracts DICOM metadata & pixels, applies correct windowing, and routes
    the study through the full multi-layer AI pipeline.
    """
    is_dicom = file.content_type == "application/dicom" or (
        file.filename and file.filename.lower().endswith(".dcm")
    )
    if not file.content_type.startswith("image/") and not is_dicom:
        raise HTTPException(status_code=400, detail="File must be an image or DICOM.")

    try:
        image_bytes = await file.read()

        # ── Server-side file size guard ───────────────────────────────────────
        if len(image_bytes) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")

        mime_type       = file.content_type
        extracted_slices: int | None = None
        gemini_payload: bytes | list[bytes] = image_bytes
        quality_frame: bytes = image_bytes
        dicom_metadata: dict = {}

        # ── DICOM path ────────────────────────────────────────────────────────
        if is_dicom:
            try:
                dicom_metadata, frames_bytes, quality_frame = read_dicom_bytes(image_bytes)
                extracted_slices = int(dicom_metadata["number_of_frames"])
                gemini_payload   = frames_bytes if len(frames_bytes) > 1 else frames_bytes[0]
                mime_type        = "image/png"
            except Exception as dcm_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to process DICOM file: {dcm_err}",
                )

        # ── Build enriched patient context for Gemini (Layer 1) ──────────────
        context_parts: list[str] = []

        resolved_age = age.strip() or dicom_metadata.get("patient_age", "")
        resolved_sex = sex.strip() or dicom_metadata.get("patient_sex", "")

        if resolved_age:
            context_parts.append(f"Age: {resolved_age}")
        if resolved_sex:
            context_parts.append(f"Sex: {resolved_sex}")
        if symptoms.strip():
            context_parts.append(f"Symptoms: {symptoms.strip()}")

        # Include DICOM study/series context for Gemini
        for key, label in (
            ("study_description",  "Study Description"),
            ("series_description", "Series Description"),
            ("body_part",          "Body Part (from DICOM)"),
            ("modality",           "Modality (from DICOM)"),
        ):
            val = dicom_metadata.get(key, "")
            if val and val != "Unknown":
                context_parts.append(f"{label}: {val}")

        patient_context = "\n".join(context_parts) if context_parts else "None provided."

        # ── Layer 1–3: Gemini generalist analysis ─────────────────────────────
        # analyze_medical_image is a sync function — run it in a thread pool
        # so it never blocks FastAPI's async event loop.
        result: dict = await asyncio.to_thread(
            analyze_medical_image, gemini_payload, mime_type, patient_context
        )

        # Slice count
        if is_dicom and extracted_slices is not None:
            result["number_of_slices"] = extracted_slices
        else:
            result["number_of_slices"] = count_grid_slices(quality_frame)

        # Objective OpenCV quality metrics
        result["objective_quality_metrics"] = assess_objective_quality(quality_frame)

        # Expose DICOM header in response for transparency
        if dicom_metadata:
            result["dicom_metadata"] = dicom_metadata

        # ── Layers 4–5: Domain Routing & Specialist Pipeline ──────────────────
        recommended_pipeline = result.get("recommended_pipeline", "")
        layer3_findings = [
            f.get("finding", "") for f in result.get("findings", []) if f.get("finding")
        ]

        pipeline_output = None
        try:
            pipeline_output = domain_router.route(
                recommended_pipeline=recommended_pipeline,
                frames=gemini_payload,
                mime_type=mime_type,
                layer3_findings=layer3_findings,
                patient_context=patient_context,
            )
            result["specialist_pipeline"] = pipeline_output.model_dump()
        except Exception as pipeline_err:
            result["specialist_pipeline"] = {
                "pipeline_name": recommended_pipeline,
                "error": str(pipeline_err),
                "key_findings": layer3_findings,
            }

        # ── Layers 6–7: Evidence Collection (Qdrant + Neo4j) ─────────────────
        try:
            retrieval_queries = (
                pipeline_output.retrieval_queries
                if pipeline_output is not None
                else layer3_findings
            )
            historical_evidence, graph_knowledge = await asyncio.gather(
                search_evidence(retrieval_queries),
                query_knowledge_graph(retrieval_queries),
            )
            result["historical_evidence"] = historical_evidence
            result["graph_knowledge"]     = graph_knowledge
        except Exception as db_err:
            print(f"Database Retrieval Error: {db_err}")
            result["historical_evidence"] = []
            result["graph_knowledge"]     = []

        # ── Layers 8–10: Inference Engine, Safety, and Audit ─────────────────
        try:
            raw_final_report = await synthesize_final_report(
                generalist_data=result,
                specialist_data=result.get("specialist_pipeline", {}),
                historical_cases=result.get("historical_evidence", []),
                graph_guidelines=result.get("graph_knowledge", []),
            )

            safe_final_report = verify_report_safety(raw_final_report)
            result["final_diagnosis_report"] = safe_final_report

            case_id = str(uuid.uuid4())
            result["case_id"] = case_id

            asyncio.create_task(
                log_case(
                    case_id=case_id,
                    request_data={
                        "modality": dicom_metadata.get("modality") if is_dicom else "Unknown",
                        "anatomy":  patient_context,
                        "image_base64": True,
                    },
                    final_report=safe_final_report,
                    safety_status=safe_final_report.get("safety_verified", False),
                )
            )
        except Exception as engine_err:
            print(f"Inference Engine Error: {engine_err}")
            result["final_diagnosis_report"] = {"error": "Failed to generate final report"}

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/grid-preview")
async def grid_preview(file: UploadFile = File(...)):
    """Returns the uploaded image with detected slice grid lines drawn on it (PNG)."""
    is_dicom = file.content_type == "application/dicom" or (
        file.filename and file.filename.lower().endswith(".dcm")
    )
    if not file.content_type.startswith("image/") and not is_dicom:
        raise HTTPException(status_code=400, detail="File must be an image or DICOM.")
    try:
        image_bytes = await file.read()
        if len(image_bytes) > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="File exceeds the 50 MB limit.")
        annotated_png = draw_grid_on_image(image_bytes)
        return Response(content=annotated_png, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
