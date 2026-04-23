from fastapi import FastAPI, UploadFile, File, HTTPException
from src.predict import VehicleDetector
import shutil
import os
import uuid

app = FastAPI(
    title="API de Detección de Vehículos",
    description="Detecta vehículos en imágenes usando YOLOv8",
    version="1.0.0"
)

detector = VehicleDetector()

@app.get("/")
def read_root():
    return {"message": "API de Detección de Vehículos funcionando"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
async def predict_vehicle(file: UploadFile = File(...)):
    # Validar que sea una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen")
    
    # Usar nombre único para evitar conflictos
    temp_path = f"temp_{uuid.uuid4().hex}_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        results = detector.detect(temp_path)
        return {
            "filename": file.filename,
            "num_detections": len(results),
            "detections": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Siempre limpiar aunque haya error
        if os.path.exists(temp_path):
            os.remove(temp_path)