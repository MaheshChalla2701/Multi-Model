from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import io
import pydicom
import numpy as np
from PIL import Image
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
        
        if is_dicom:
            try:
                # Load DICOM from bytes
                dicom_ds = pydicom.dcmread(io.BytesIO(image_bytes))
                
                # Extract number of slices
                extracted_slices = int(getattr(dicom_ds, "NumberOfFrames", 1))
                
                # Get pixel array
                pixel_array = dicom_ds.pixel_array
                
                # If multi-frame, take the middle frame
                if len(pixel_array.shape) > 2 and extracted_slices > 1:
                    middle_idx = extracted_slices // 2
                    pixel_array = pixel_array[middle_idx]
                
                # Normalize to 0-255 for standard image conversion
                if np.max(pixel_array) > 0:
                    pixel_array = pixel_array - np.min(pixel_array)
                    pixel_array = (pixel_array / np.max(pixel_array)) * 255.0
                pixel_array = pixel_array.astype(np.uint8)
                
                # Convert to PNG
                image = Image.fromarray(pixel_array)
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                image_bytes = img_byte_arr.getvalue()
                mime_type = "image/png"
            except Exception as dcm_err:
                raise HTTPException(status_code=400, detail=f"Failed to process DICOM file: {str(dcm_err)}")

        result = analyze_medical_image(image_bytes, mime_type, patient_context)
        
        # Override the slice count with exact DICOM metadata if available
        if is_dicom and extracted_slices is not None:
            result["number_of_slices"] = extracted_slices
            
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "ok", "service": "Medical Understanding API"}
