"""
Smart Auto Crop AI - Main FastAPI Application
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from typing import Optional
import time
import traceback

from utils.image_processor import ImageProcessor
from services.face_detector import FaceDetector
from services.object_detector import ObjectDetector
from services.saliency_detector import SaliencyDetector
from services.crop_analyzer import CropAnalyzer

# Initialize FastAPI app
app = FastAPI(
    title="Smart Auto Crop AI",
    description="AI-powered intelligent image cropping service",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
face_detector = FaceDetector()
object_detector = ObjectDetector()
saliency_detector = SaliencyDetector()
crop_analyzer = CropAnalyzer()
image_processor = ImageProcessor()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page"""
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Smart Auto Crop AI"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    """
    Analyze uploaded image and return smart crop suggestions
    
    Args:
        file: Uploaded image file
        
    Returns:
        JSON response with crop analysis results
    """
    start_time = time.time()
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate image
        is_valid, error_msg = image_processor.validate_image(file_content, file.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        # Convert to OpenCV format
        img = image_processor.bytes_to_cv2(file_content)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        original_height, original_width = img.shape[:2]
        
        # Resize for faster processing
        img_resized, scale = image_processor.resize_for_processing(img, max_dimension=1024)
        
        # Step 1: Face Detection
        print("Detecting faces...")
        faces = face_detector.detect_faces(img_resized)
        largest_face = face_detector.get_largest_face(faces) if faces else None
        
        # Step 2: Object Detection
        print("Detecting objects...")
        objects = object_detector.detect_objects(img_resized, confidence=0.3)
        important_objects = object_detector.filter_important_objects(objects)
        primary_object = object_detector.get_primary_object(important_objects) if important_objects else None
        
        # Step 3: Saliency Detection
        print("Analyzing saliency...")
        success, saliency_map = saliency_detector.detect_saliency(img_resized)
        saliency_center = saliency_detector.get_saliency_center(saliency_map)
        
        # Determine primary subject (face > object > saliency)
        subject_bbox = None
        subject_type = "saliency"
        
        if largest_face:
            subject_bbox = largest_face
            subject_type = "face"
        elif primary_object:
            subject_bbox = primary_object['bbox']
            subject_type = f"object ({primary_object['class']})"
        
        # Step 4: Generate crop candidates
        print("Generating crop candidates...")
        candidates = crop_analyzer.generate_crop_candidates(
            img_resized.shape[1],
            img_resized.shape[0],
            subject_bbox,
            saliency_center
        )
        
        # Step 5: Score each candidate
        print("Scoring crops...")
        scored_candidates = []
        for candidate in candidates:
            scored = crop_analyzer.score_crop(
                candidate,
                img_resized,
                subject_bbox,
                saliency_map
            )
            scored_candidates.append(scored)
        
        # Sort by total score
        scored_candidates.sort(key=lambda c: c.total_score, reverse=True)
        best_candidate = scored_candidates[0]
        
        # Step 6: Generate cropped image
        print("Generating cropped image...")
        cropped_img = image_processor.crop_image(
            img_resized,
            best_candidate.x,
            best_candidate.y,
            best_candidate.width,
            best_candidate.height
        )
        
        # Convert to base64
        cropped_base64 = image_processor.cv2_to_base64(cropped_img, format='JPEG')
        original_base64 = image_processor.cv2_to_base64(img_resized, format='JPEG')
        
        # Draw crop box on original for visualization
        img_with_box = image_processor.draw_crop_box(
            img_resized,
            best_candidate.x,
            best_candidate.y,
            best_candidate.width,
            best_candidate.height,
            color=(0, 255, 0),
            thickness=3
        )
        preview_base64 = image_processor.cv2_to_base64(img_with_box, format='JPEG')
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response = {
            "success": True,
            "processing_time": round(processing_time, 2),
            "original_dimensions": {
                "width": original_width,
                "height": original_height
            },
            "cropped_image": cropped_base64,
            "original_image": original_base64,
            "preview_image": preview_base64,
            "crop_box": {
                "x": int(best_candidate.x / scale),
                "y": int(best_candidate.y / scale),
                "width": int(best_candidate.width / scale),
                "height": int(best_candidate.height / scale)
            },
            "aspect_ratio": best_candidate.aspect_ratio,
            "scores": {
                "subject_score": round(best_candidate.subject_score, 3),
                "composition_score": round(best_candidate.composition_score, 3),
                "balance_score": round(best_candidate.balance_score, 3),
                "saliency_score": round(best_candidate.saliency_score, 3),
                "total_score": round(best_candidate.total_score, 3)
            },
            "explanation": best_candidate.explanation,
            "analysis": {
                "subject_type": subject_type,
                "faces_detected": len(faces),
                "objects_detected": len(objects),
                "important_objects": len(important_objects)
            },
            "all_candidates": [
                {
                    "aspect_ratio": c.aspect_ratio,
                    "total_score": round(c.total_score, 3),
                    "explanation": c.explanation
                }
                for c in scored_candidates
            ]
        }
        
        print(f"Analysis complete in {processing_time:.2f}s")
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    print("Starting Smart Auto Crop AI server...")
    print("Server will be available at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
