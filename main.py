from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os

from model_utils import (
    DISEASE_CLASSES, preprocess_image, 
    load_disease_model, load_crop_model
)
from schemas import CropRecommendRequest

app = FastAPI(title="Krisharthi ML Backend", version="2.1.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
MODEL_DIR = "models"
DISEASE_MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_recommendation_model.pkl")

# Load models at startup
disease_model = load_disease_model(DISEASE_MODEL_PATH)
crop_model = load_crop_model(CROP_MODEL_PATH)

@app.get("/api/ml/health")
def read_health():
    return {
        "status": "ok",
        "models": {
            "disease_classifier": disease_model is not None,
            "crop_recommender": crop_model is not None
        }
    }

@app.post("/api/ml/diagnose")
async def diagnose_crop(image: UploadFile = File(...)):
    """
    Accepts an image file, returns plant disease classification result.
    The Main Backend will use this result to generate natural language advice.
    """
    if not disease_model:
        raise HTTPException(status_code=503, detail="Disease detection model not loaded on server.")
    
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        processed_img = preprocess_image(img)
        
        prediction = disease_model.predict(processed_img)
        result_idx = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        disease_name = DISEASE_CLASSES[result_idx]
        
        return {
            "success": True,
            "data": {
                "diseaseName": disease_name,
                "confidence": round(confidence * 100, 2),
                "timestamp": float(tf.timestamp().numpy())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/api/ml/recommend-crop")
def recommend_crop(data: CropRecommendRequest):
    """
    Accepts soil and weather parameters, returns recommended crop.
    The Main Backend will use this result to generate natural language advice.
    """
    if not crop_model:
        raise HTTPException(status_code=503, detail="Crop recommendation model not loaded on server.")
    
    try:
        input_data = np.array([[
            data.nitrogen, data.phosphorus, data.potassium, 
            data.temperature, data.humidity, data.ph, data.rainfall
        ]])
        
        prediction = crop_model.predict(input_data)
        recommended = prediction[0]
        
        return {
            "success": True,
            "data": {
                "recommendedCrop": recommended.capitalize(),
                "inputParams": data.model_dump()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
