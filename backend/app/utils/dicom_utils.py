"""
dicom_utils.py
--------------
All DICOM-specific processing helpers.

Responsibilities:
  - Window/Level presets (Hounsfield Unit → 0-255 mapping)
  - DICOM header metadata extraction
  - Single-frame pixel → PNG bytes conversion
"""

import io
import numpy as np
import pydicom
import cv2


# ── DICOM Windowing Presets ──────────────────────────────────────────────────
# Maps anatomy/modality keywords → (WindowCenter, WindowWidth)
# These are standard radiological window/level presets for correct rendering.
WINDOW_PRESETS: dict[str, tuple[int, int]] = {
    "brain":       (40,    80),
    "subdural":    (75,   215),
    "stroke":      (40,    40),
    "bone":        (400, 1800),
    "lung":        (-600, 1500),
    "chest":       (-600, 1500),
    "mediastinum": (40,   400),
    "abdomen":     (60,   400),
    "liver":       (60,   160),
    "spine":       (400, 1800),
    "default":     (40,   400),
}


def get_window_preset(
    study_desc: str,
    body_part: str,
    modality: str,
) -> tuple[int, int] | None:
    """
    Selects the best Window/Level preset based on available DICOM metadata.
    Checks study description and body part fields for known keywords.
    Returns None for plain X-rays (CR/DX/RG) — those use min-max normalisation.
    """
    combined = f"{study_desc} {body_part}".lower()
    for keyword, preset in WINDOW_PRESETS.items():
        if keyword in combined:
            return preset
    # Plain X-ray modalities don't need HU windowing
    if modality in ("CR", "DX", "RG"):
        return None
    return WINDOW_PRESETS["default"]


def apply_window(
    pixel_array: np.ndarray,
    window_center: int,
    window_width: int,
) -> np.ndarray:
    """
    Applies standard radiological Window/Level transform to a 2-D pixel array.
    Converts Hounsfield Units → 0-255 uint8 for display.
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(pixel_array, lower, upper)
    windowed = ((windowed - lower) / (upper - lower) * 255.0).astype(np.uint8)
    return windowed


def normalize_generic(pixel_array: np.ndarray) -> np.ndarray:
    """
    Generic min-max normalisation.
    Used when no window preset is available (e.g., plain X-rays where raw pixel
    values already represent optical density).
    """
    arr = pixel_array.astype(np.float32)
    min_val, max_val = arr.min(), arr.max()
    if max_val > min_val:
        arr = (arr - min_val) / (max_val - min_val) * 255.0
    return arr.astype(np.uint8)


def extract_dicom_metadata(dicom_ds) -> dict:
    """
    Safely extracts all clinically useful DICOM header tags.
    Returns a dict of available metadata fields.
    """
    def safe_get(tag: str, default: str = "Unknown") -> str:
        val = getattr(dicom_ds, tag, None)
        if val is None or str(val).strip() == "":
            return default
        return str(val).strip()

    return {
        "modality":           safe_get("Modality"),
        "body_part":          safe_get("BodyPartExamined"),
        "study_description":  safe_get("StudyDescription"),
        "series_description": safe_get("SeriesDescription"),
        "patient_age":        safe_get("PatientAge"),
        "patient_sex":        safe_get("PatientSex"),
        "institution":        safe_get("InstitutionName"),
        "manufacturer":       safe_get("Manufacturer"),
        "rows":               safe_get("Rows"),
        "columns":            safe_get("Columns"),
        "number_of_frames":   safe_get("NumberOfFrames", "1"),
        # DICOM-embedded window values (may or may not exist)
        "window_center":      safe_get("WindowCenter"),
        "window_width":       safe_get("WindowWidth"),
    }


def process_dicom_frame(
    frame: np.ndarray,
    window_preset: tuple[int, int] | None,
) -> bytes:
    """
    Converts a single raw DICOM frame (2-D array) → windowed PNG bytes.
    """
    if window_preset is not None:
        wc, ww = window_preset
        normalized = apply_window(frame, wc, ww)
    else:
        normalized = normalize_generic(frame)

    _, img_buf = cv2.imencode(".png", normalized)
    return img_buf.tobytes()


def read_dicom_bytes(
    image_bytes: bytes,
) -> tuple[dict, list[bytes], bytes]:
    """
    High-level helper used by the API endpoint.

    Reads raw DICOM bytes and returns:
      - dicom_metadata  : extracted header dict
      - gemini_frames   : list of PNG bytes (1 item for single-frame, up to 10 for multi-frame)
      - quality_frame   : the middle frame PNG bytes (used for quality metrics)

    Raises ValueError on parse failure.
    """
    dicom_ds = pydicom.dcmread(io.BytesIO(image_bytes))
    dicom_metadata = extract_dicom_metadata(dicom_ds)
    num_frames = int(dicom_metadata["number_of_frames"])

    # Resolve windowing preset
    wc_raw = dicom_metadata["window_center"]
    ww_raw = dicom_metadata["window_width"]
    if wc_raw != "Unknown" and ww_raw != "Unknown":
        try:
            wc = int(float(str(wc_raw).split("\\")[0]))
            ww = int(float(str(ww_raw).split("\\")[0]))
            window_preset: tuple[int, int] | None = (wc, ww)
        except (ValueError, IndexError):
            window_preset = get_window_preset(
                dicom_metadata["study_description"],
                dicom_metadata["body_part"],
                dicom_metadata["modality"],
            )
    else:
        window_preset = get_window_preset(
            dicom_metadata["study_description"],
            dicom_metadata["body_part"],
            dicom_metadata["modality"],
        )

    pixel_array = dicom_ds.pixel_array

    if len(pixel_array.shape) > 2 and num_frames > 1:
        # Multi-frame DICOM: sample up to 10 evenly spaced frames
        num_samples = min(num_frames, 10)
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)
        frames_bytes = [process_dicom_frame(pixel_array[idx], window_preset) for idx in indices]
        quality_frame = frames_bytes[len(frames_bytes) // 2]
    else:
        frame_2d = pixel_array if len(pixel_array.shape) == 2 else pixel_array[0]
        single = process_dicom_frame(frame_2d, window_preset)
        frames_bytes = [single]
        quality_frame = single

    return dicom_metadata, frames_bytes, quality_frame
