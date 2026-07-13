# Enervara — Medical Imaging Analysis Platform

A production-grade, multi-layer AI diagnostic platform that processes X-Ray, MRI, CT, Ultrasound, and DICOM files. The system transforms raw medical images and optional patient context into structured, evidence-based radiological findings by routing each case through a chain of specialist AI pipelines backed by vector search and a clinical knowledge graph.

> **Disclaimer:** This platform is an AI research tool. All outputs are informational only and must not be used as a substitute for professional medical advice or diagnosis.

---

## Architecture

The pipeline has **10 layers**, executed in sequence for each uploaded scan:

```
Image Upload (DICOM or standard image)
        │
        ▼
┌─────────────────────────────────────────────┐
│  Layer 1–3 · Generalist Analysis (Gemini)   │
│  Quality Check → Modality/Anatomy ID        │
│  → Finding Extraction → Calibration         │
└────────────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Layer 4 · Domain Router│
        └────────────┬────────────┘
                     │  routes to one of 19 specialist pipelines
        ┌────────────▼──────────────────────────┐
        │  Layer 5 · Specialist Pipeline (Gemini)│
        │  Chest X-Ray · Neuro MRI · Trauma CT  │
        │  MSK MRI · Spine MRI · Abdomen CT …   │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Layer 6 · Evidence Collection (Qdrant)│
        │  Vector similarity search on findings  │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Layer 7 · Knowledge Graph (Neo4j)     │
        │  Finding → Disease → Guideline lookup  │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼──────────────────────────┐
        │  Layer 8 · Inference Engine (Gemini)   │
        │  Synthesises final diagnostic report   │
        └────────────┬──────────────────────────┘
                     │
        ┌────────────▼────────────────────────┐
        │  Layer 9 · Safety & Compliance       │
        │  Adds disclaimer + urgency alerts    │
        └────────────┬────────────────────────┘
                     │
        ┌────────────▼────────────────────────┐
        │  Layer 10 · Audit Log (MongoDB)      │
        │  Async case record for traceability  │
        └─────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 16, Tailwind CSS v4 |
| **Backend** | Python 3.10+, FastAPI 0.115 |
| **AI Engine** | Google Gemini 2.5 Flash (`google-genai`) |
| **DICOM Processing** | `pydicom` + `opencv-python-headless` + `numpy` |
| **Vector DB** | Qdrant Cloud (historical case search) |
| **Knowledge Graph** | Neo4j AuraDB (Finding → Disease → Guideline) |
| **Audit DB** | MongoDB (async case logging) |
| **Testing** | `pytest` + `pytest-asyncio` + `httpx` |

---

## Specialist Pipelines

19 domain-specific pipelines cover the most common radiology workloads:

| Modality | Pipelines |
|---|---|
| X-Ray | Chest X-Ray, Bone X-Ray, Dental X-Ray, Mammography |
| MRI | Neuro MRI, Spine MRI, MSK MRI, Abdominal MRI, Cardiac MRI |
| CT | Neuro CT, Chest CT, Abdomen CT, Trauma CT |
| Ultrasound | Fetal Ultrasound |
| Generic fallbacks | Generic X-Ray, Generic CT, Generic MRI, Generic Ultrasound |

Unknown modalities fall back to the generic pipeline most appropriate for their name, or a minimal pass-through if unrecognised.

---

## Project Structure

```
Multi-Model/
├── frontend/                       # Next.js application
│   ├── src/app/
│   │   ├── page.tsx                # Single-page UI (upload + results)
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── .env.local.example          # Copy → .env.local and set API URL
│   └── package.json
│
└── backend/                        # FastAPI application
    ├── app/
    │   ├── main.py                 # API routes (analyze-scan, grid-preview)
    │   ├── models/
    │   │   ├── schemas.py          # Pydantic models for Gemini extraction
    │   │   └── pipeline_schemas.py # PipelineOutput / AbnormalRegion models
    │   ├── services/
    │   │   ├── gemini_service.py   # Layer 1–3: generalist Gemini analysis
    │   │   ├── calibration.py      # Ordinal confidence → float score
    │   │   ├── router.py           # Layer 4: domain router (19 pipelines)
    │   │   ├── qdrant_service.py   # Layer 6: vector evidence search
    │   │   ├── neo4j_service.py    # Layer 7: knowledge graph query
    │   │   ├── inference_engine.py # Layer 8: CMO synthesis report
    │   │   ├── safety_service.py   # Layer 9: disclaimer + urgency flags
    │   │   └── audit_service.py    # Layer 10: async MongoDB audit log
    │   ├── pipelines/
    │   │   ├── base.py             # BasePipeline ABC (shared Gemini engine)
    │   │   ├── chest_xray.py       # … (19 specialist pipeline files)
    │   │   └── …
    │   └── utils/
    │       ├── dicom_utils.py      # DICOM windowing, metadata, frame extraction
    │       └── slicing.py          # CV quality metrics + grid-panel detection
    ├── scripts/
    │   ├── seed_qdrant.py          # Populate Qdrant with sample cases
    │   └── seed_neo4j.py           # Populate Neo4j with clinical knowledge graph
    ├── tests/
    │   ├── conftest.py             # Env var seeding for isolated test runs
    │   ├── test_api.py             # Integration tests (FastAPI + httpx)
    │   ├── test_calibration.py     # Unit tests for calibration layer
    │   └── test_dicom_utils.py     # Unit tests for windowing/normalisation
    ├── .env.example                # Copy → .env and fill in credentials
    ├── requirements.txt            # Pinned Python dependencies
    └── pytest.ini                  # Pytest configuration (asyncio mode)
```

---

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+
- A **Google Gemini API key** (required — get one at [aistudio.google.com](https://aistudio.google.com))
- *(Optional)* Qdrant Cloud, Neo4j AuraDB, and MongoDB credentials for full pipeline. The app runs gracefully in **mock mode** without them.

---

### 1. Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies (all versions pinned)
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and set at minimum: GEMINI_API_KEY
```

**`.env` reference:**

```env
GEMINI_API_KEY="your_google_ai_studio_key_here"

# Optional — the app works without these (mock mode is used instead)
QDRANT_URL="https://your-cluster.cloud.qdrant.io"
QDRANT_API_KEY="your_qdrant_key"
NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="your_neo4j_password"
MONGODB_URL="mongodb://localhost:27017"

# CORS — restrict to your frontend origin in production
ALLOWED_ORIGINS="http://localhost:3000"
```

```bash
# Start the API server
uvicorn app.main:app --reload
# API is now running at http://localhost:8000
```

---

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# (Optional) Configure backend URL
cp .env.local.example .env.local
# Edit .env.local: NEXT_PUBLIC_API_URL=http://localhost:8000

# Start the dev server
npm run dev
# Frontend is now running at http://localhost:3000
```

---

### 3. (Optional) Seed the Databases

If you have Qdrant and Neo4j configured, seed them with sample data to enable real evidence retrieval:

```bash
cd backend

# Seed Qdrant with 4 sample historical cases
python scripts/seed_qdrant.py

# Seed Neo4j with the clinical knowledge graph
# (Finding → Disease → Symptom / Guideline nodes)
python scripts/seed_neo4j.py
```

---

### 4. Access the Application

Open `http://localhost:3000`. Upload any medical scan (X-Ray, MRI, CT, Ultrasound, or a `.dcm` DICOM file) and optionally provide patient context (age, sex, symptoms).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/api/analyze-scan` | Full 10-layer analysis pipeline |
| `POST` | `/api/grid-preview` | Returns annotated PNG with detected slice grid |

**`POST /api/analyze-scan` — form fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | Image (`image/*`) or DICOM (`.dcm`) — max 50 MB |
| `age` | string | ❌ | Patient age (auto-extracted from DICOM if available) |
| `sex` | string | ❌ | Patient sex |
| `symptoms` | string | ❌ | Free-text clinical notes |

---

## Running Tests

```bash
cd backend
venv\Scripts\python -m pytest tests/ -v   # Windows
# or
python -m pytest tests/ -v               # Mac/Linux (with venv active)
```

Current test coverage: **42 tests** across 3 modules.

| Module | Tests | Coverage |
|---|---|---|
| `test_api.py` | 6 | Health check, file validation, 50 MB guard, full happy-path (all external services mocked) |
| `test_calibration.py` | 10 | Confidence map integrity, score ordering, extraction output structure |
| `test_dicom_utils.py` | 26 | Windowing, normalisation, preset selection (all 10 presets) |

---

## DICOM Support

The backend provides full DICOM handling:

- **Windowing**: Automatically applies correct Hounsfield Unit → 0-255 mapping using 10 standard presets (`brain`, `lung`, `bone`, `abdomen`, etc.), preferring embedded DICOM window values when present
- **Multi-frame**: Samples up to 10 evenly-spaced frames from multi-frame DICOMs (e.g., CT volumes) and sends all frames to Gemini for analysis
- **Metadata extraction**: Modality, body part, study/series description, patient age/sex are automatically extracted from DICOM headers and used to enrich the Gemini prompt
- **Grid preview**: OpenCV-based slice-panel detection with annotated PNG overlay

---

## Gradual Feature Enablement

The system is designed so each external service degrades gracefully:

| Service | Without credentials | With credentials |
|---|---|---|
| Qdrant | Returns 2 mock evidence records | Real vector similarity search |
| Neo4j | Returns mock disease/guideline records | Real Cypher graph queries |
| MongoDB | Logs to console only | Persistent audit trail |
| Gemini | Hard failure (required) | Full structured analysis |
