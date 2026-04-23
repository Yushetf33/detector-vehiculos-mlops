from ultralytics import YOLO
import os

class VehicleDetector:
    # Clases de vehículos en COCO dataset (usadas por YOLOv8)
    VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}

    def __init__(self, model_path='models/yolov8n.pt'):
        if not os.path.exists(model_path):
            print(f"Modelo no encontrado en {model_path}, descargando YOLOv8n...")
        self.model = YOLO(model_path)

    def detect(self, image_path: str, confidence_threshold: float = 0.5):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Imagen no encontrada: {image_path}")
        
        results = self.model(image_path)
        detections = []
        
        for r in results:
            for box in r.boxes:
                class_name = r.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                
                # Filtrar solo vehículos y por confianza mínima
                if class_name in self.VEHICLE_CLASSES and confidence >= confidence_threshold:
                    detections.append({
                        "class": class_name,
                        "confidence": round(confidence, 3),
                        "bbox": [round(x, 2) for x in box.xyxy[0].tolist()]
                    })
        
        return detections