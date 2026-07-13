"""
slicing.py
----------
OpenCV-based helpers for medical scan collage / slice-grid analysis.

Responsibilities:
  - Objective image quality metrics (blur, contrast, brightness)
  - Grid-panel detection via projection profiles + run-length encoding
  - Annotated grid-overlay image generation
"""

import cv2
import numpy as np


def assess_objective_quality(image_bytes: bytes) -> dict:
    """
    Calculates objective image quality metrics using OpenCV.

    Returns:
        blur_score                 – Variance of Laplacian. <100 typically means blurry.
        contrast_score             – Std-dev of pixel intensities. <30 = flat/washed-out.
        brightness_score           – Mean pixel intensity (0-255).
        is_mathematically_blurry   – True when blur_score < 100.
        is_mathematically_low_contrast – True when contrast_score < 30.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {}

        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        mean, stddev = cv2.meanStdDev(img)
        contrast_score = float(stddev[0][0])
        brightness_score = float(mean[0][0])

        return {
            "blur_score":                    round(blur_score, 2),
            "contrast_score":                round(contrast_score, 2),
            "brightness_score":              round(brightness_score, 2),
            "is_mathematically_blurry":      bool(blur_score < 100),
            "is_mathematically_low_contrast": bool(contrast_score < 30),
        }
    except Exception:
        return {}


def _get_segments(proj: np.ndarray) -> list[tuple[int, int, bool]]:
    """
    Returns a list of (start, end, is_separator) tuples for each content run.

    Uses a RELATIVE threshold: a row/col is treated as a separator if its
    brightness is in the lowest 15% of the observed brightness range.
    This adapts automatically to dark-background (black film) and
    coloured-background (blue/teal) MRI collages.
    """
    total = len(proj)
    min_sep     = max(1, int(total * 0.005))   # separator >= 0.5 % of axis
    min_content = max(2, int(total * 0.05))    # content panel >= 5.0 % of axis

    lo, hi = float(np.min(proj)), float(np.max(proj))
    sep_thresh = lo + (hi - lo) * 0.15 if hi > lo else lo

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
    result: list[tuple[int, int, bool]] = []
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
      - Green lines  = separator midpoints
      - Cyan boxes   = content panels

    Returns a high-resolution PNG.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")

    h, w = img.shape[:2]

    # Upscale to at least 900 px on the longest side for crisp display
    scale = max(1.0, 900 / max(h, w))
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

    # Horizontal separator lines (green)
    for start, end, is_sep in row_segs:
        if is_sep:
            mid = (start + end) // 2
            cv2.line(annotated, (0, mid), (w, mid), (0, 220, 80), line_w)

    # Vertical separator lines (green)
    for start, end, is_sep in col_segs:
        if is_sep:
            mid = (start + end) // 2
            cv2.line(annotated, (mid, 0), (mid, h), (0, 220, 80), line_w)

    # Cyan bounding boxes around each detected content panel
    content_rows = [(s, e) for s, e, sep in row_segs if not sep]
    content_cols = [(s, e) for s, e, sep in col_segs if not sep]
    for rs, re in content_rows:
        for cs, ce in content_cols:
            cv2.rectangle(annotated, (cs, rs), (ce - 1, re - 1), (0, 220, 220), line_w)

    # Slice count label in top-left
    total = max(1, len(content_rows) * len(content_cols))
    font_scale = max(0.5, min(h, w) / 500)
    cv2.putText(
        annotated,
        f"Slices: {total}",
        (max(6, line_w * 2), int(26 * font_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 255),
        max(1, line_w),
        cv2.LINE_AA,
    )

    _, buf = cv2.imencode(".png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    return buf.tobytes()
