from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import pydicom
import numpy as np

from app.services.gemini_service import analyze_medical_image

app = FastAPI(title="Medical Understanding API")

# Allow CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import cv2

def assess_objective_quality(image_bytes: bytes) -> dict:
    """
    Calculates objective image quality metrics using OpenCV.
    - Blur Score: Variance of Laplacian. Lower = blurrier (typically < 100 is blurry).
    - Contrast Score: Standard deviation of pixel intensities. Lower = flat/washed out.
    - Brightness: Mean pixel intensity (0-255).
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {}

        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        mean, stddev = cv2.meanStdDev(img)
        contrast_score = stddev[0][0]
        brightness_score = mean[0][0]

        return {
            "blur_score": round(blur_score, 2),
            "contrast_score": round(contrast_score, 2),
            "brightness_score": round(brightness_score, 2),
            "is_mathematically_blurry": bool(blur_score < 100),
            "is_mathematically_low_contrast": bool(contrast_score < 30)
        }
    except Exception:
        return {}


def _get_segments(proj: np.ndarray) -> list[tuple[int, int, bool]]:
    """
    Returns a list of (start, end, is_separator) tuples for each content run.

    Uses a RELATIVE threshold: a row/col is treated as a separator if its
    brightness is in the lowest 30% of the observed brightness range.
    This adapts automatically to dark-background (black film) and
    colored-background (blue/teal) MRI collages.
    """
    total = len(proj)
    min_sep     = max(1, int(total * 0.005))   # separator ≥ 0.5% of axis
    min_content = max(2, int(total * 0.05))    # content panel ≥ 5.0% of axis

    lo, hi = float(np.min(proj)), float(np.max(proj))
    # Relative separator threshold: bottom 15% of brightness range
    if hi > lo:
        sep_thresh = lo + (hi - lo) * 0.15
    else:
        sep_thresh = lo  # uniform image — nothing to separate

    is_sep = proj <= sep_thresh

    # Run-length encode
    runs: list[tuple[bool, int]] = []
    cur, cnt = bool(is_sep[0]), 1
    for val in is_sep[1:]:
        if bool(val) == cur:
            cnt += 1
        else:
            runs.append((cur, cnt))
            cur, cnt = bool(val), 1
    runs.append((cur, cnt))

    # Convert to (start, end, is_sep) and filter noise
    result = []
    pos = 0
    for sep, ln in runs:
        end = pos + ln
        if (sep and ln >= min_sep) or (not sep and ln >= min_content):
            result.append((pos, end, sep))
        pos = end
    return result



def count_grid_slices(image_bytes: bytes) -> int:
    """
    Counts image panels in a medical scan collage using OpenCV.
    Uses Otsu thresholding + projection profiles + run-length encoding.
    Returns 1 for a single-panel image.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 1

        h, w = img.shape
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        row_proj = binary.sum(axis=1).astype(float) / w
        col_proj = binary.sum(axis=0).astype(float) / h

        rows = max(1, sum(1 for _, _, sep in _get_segments(row_proj) if not sep))
        cols = max(1, sum(1 for _, _, sep in _get_segments(col_proj) if not sep))

        return max(1, rows * cols)
    except Exception:
        return 1


def draw_grid_on_image(image_bytes: bytes) -> bytes:
    """
    Upscales the image, then draws grid lines on detected slice boundaries.
    Green lines = separator midpoints, cyan boxes = content panels.
    Returns a high-resolution PNG.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]

    # Upscale to at least 900px on the longest side for crisp display
    target = 900
    scale = max(1.0, target / max(h, w))
    if scale > 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        h, w = img.shape[:2]

    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    row_segs = _get_segments(binary.sum(axis=1).astype(float) / w)
    col_segs = _get_segments(binary.sum(axis=0).astype(float) / h)

    annotated = img.copy()
    line_w = max(1, int(min(h, w) * 0.003))   # line thickness scales with image

    # Draw horizontal separator lines in green
    for start, end, is_sep in row_segs:
        if is_sep:
            mid = (start + end) // 2
            cv2.line(annotated, (0, mid), (w, mid), (0, 220, 80), line_w)

    # Draw vertical separator lines in green
    for start, end, is_sep in col_segs:
        if is_sep:
            mid = (start + end) // 2
            cv2.line(annotated, (mid, 0), (mid, h), (0, 220, 80), line_w)

    # Draw cyan bounding boxes around each detected content panel
    content_rows = [(s, e) for s, e, sep in row_segs if not sep]
    content_cols = [(s, e) for s, e, sep in col_segs if not sep]
    for rs, re in content_rows:
        for cs, ce in content_cols:
            cv2.rectangle(annotated, (cs, rs), (ce - 1, re - 1), (0, 220, 220), line_w)

    # Label total count in top-left with readable font size
    total = max(1, len(content_rows) * len(content_cols))
    font_scale = max(0.5, min(h, w) / 500)
    cv2.putText(annotated, f"Slices: {total}", (max(6, line_w * 2), int(26 * font_scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), max(1, line_w), cv2.LINE_AA)

    _, buf = cv2.imencode(".png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()



@app.post("/api/analyze-scan")
async def analyze_scan(
    file: UploadFile = File(...),
    patient_context: str = Form("")
):
    is_dicom = file.content_type == "application/dicom" or (file.filename and file.filename.lower().endswith(".dcm"))
    if not file.content_type.startswith("image/") and not is_dicom:
        raise HTTPException(status_code=400, detail="File must be an image or DICOM.")
    
    try:
        image_bytes = await file.read()
        mime_type = file.content_type
        extracted_slices = None
        gemini_payload = image_bytes
        
        if is_dicom:
            try:
                # Load DICOM from bytes
                dicom_ds = pydicom.dcmread(io.BytesIO(image_bytes))
                
                # Extract number of slices
                extracted_slices = int(getattr(dicom_ds, "NumberOfFrames", 1))
                import cv2
                
                # Get pixel array
                pixel_array = dicom_ds.pixel_array
                
                if len(pixel_array.shape) > 2 and extracted_slices > 1:
                    # Multi-frame DICOM: Sample up to 10 evenly spaced frames
                    num_samples = min(extracted_slices, 10)
                    indices = np.linspace(0, extracted_slices - 1, num_samples, dtype=int)
                    
                    frames_bytes = []
                    for idx in indices:
                        frame = pixel_array[idx]
                        if np.max(frame) > 0:
                            frame = frame - np.min(frame)
                            frame = (frame / np.max(frame)) * 255.0
                        frame = frame.astype(np.uint8)
                        _, img_buf = cv2.imencode(".png", frame)
                        frames_bytes.append(img_buf.tobytes())
                    
                    # Pass the list of frames to Gemini
                    gemini_payload = frames_bytes
                    # Keep the middle frame for objective metrics
                    image_bytes = frames_bytes[len(frames_bytes) // 2]
                else:
                    # Single frame
                    if np.max(pixel_array) > 0:
                        pixel_array = pixel_array - np.min(pixel_array)
                        pixel_array = (pixel_array / np.max(pixel_array)) * 255.0
                    pixel_array = pixel_array.astype(np.uint8)
                    _, img_buf = cv2.imencode(".png", pixel_array)
                    image_bytes = img_buf.tobytes()
                    gemini_payload = image_bytes
                    
                mime_type = "image/png"
            except Exception as dcm_err:
                raise HTTPException(status_code=400, detail=f"Failed to process DICOM file: {str(dcm_err)}")

        result = analyze_medical_image(gemini_payload, mime_type, patient_context)
        
        # Override the slice count with exact DICOM metadata if available,
        # otherwise count programmatically from the image pixel data.
        if is_dicom and extracted_slices is not None:
            result["number_of_slices"] = extracted_slices
        else:
            result["number_of_slices"] = count_grid_slices(image_bytes)
            
        # Add objective OpenCV quality metrics
        result["objective_quality_metrics"] = assess_objective_quality(image_bytes)
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Medical Understanding API"}


@app.post("/api/grid-preview")
async def grid_preview(file: UploadFile = File(...)):
    """Returns the uploaded image with detected slice grid lines drawn on it (PNG)."""
    from fastapi.responses import Response

    is_dicom = file.content_type == "application/dicom" or (file.filename and file.filename.lower().endswith(".dcm"))
    if not file.content_type.startswith("image/") and not is_dicom:
        raise HTTPException(status_code=400, detail="File must be an image or DICOM.")
    try:
        image_bytes = await file.read()
        annotated_png = draw_grid_on_image(image_bytes)
        return Response(content=annotated_png, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
