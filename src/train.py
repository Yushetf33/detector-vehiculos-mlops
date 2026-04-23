from ultralytics import YOLO
import wandb
import os
import time

def evaluate_model(
    model_path: str = "yolov8n.pt",
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    img_size: int = 640,
    run_name: str = None,
    output_dir: str = "models"
):
    # Nombre automático si no se especifica
    model_name = model_path.replace(".pt", "")
    if run_name is None:
        run_name = f"{model_name}_conf{confidence_threshold}_iou{iou_threshold}_img{img_size}"

    wandb.init(
        project="vehicle-detector-mlops",
        name=run_name,
        config={
            "model": model_path,
            "dataset": "COCO128",
            "confidence_threshold": confidence_threshold,
            "iou_threshold": iou_threshold,
            "img_size": img_size,
            "task": "detection"
        }
    )

    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_path)

    start_time = time.time()
    results = model.val(
        data="coco128.yaml",
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=img_size
    )
    inference_time = time.time() - start_time

    wandb.log({
        "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
        "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
        "precision": results.results_dict.get("metrics/precision(B)", 0),
        "recall": results.results_dict.get("metrics/recall(B)", 0),
        "inference_time_seconds": round(inference_time, 2),
        "fitness": results.fitness,
    })

    wandb.finish()
    print(f"✅ Completado: {run_name}")
    return results


if __name__ == "__main__":

    experiments = [
        # --- Comparativa de modelos (mismos parámetros) ---
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.25, "iou_threshold": 0.7,  "img_size": 640, "run_name": "nano_baseline"},
        {"model_path": "yolov8s.pt", "confidence_threshold": 0.25, "iou_threshold": 0.7,  "img_size": 640, "run_name": "small_baseline"},
        {"model_path": "yolov8m.pt", "confidence_threshold": 0.25, "iou_threshold": 0.7,  "img_size": 640, "run_name": "medium_baseline"},

        # --- Efecto del umbral de confianza (modelo nano) ---
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.10, "iou_threshold": 0.7,  "img_size": 640, "run_name": "nano_low_confidence"},
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.50, "iou_threshold": 0.7,  "img_size": 640, "run_name": "nano_high_confidence"},

        # --- Efecto del umbral IoU (modelo nano) ---
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.25, "iou_threshold": 0.5,  "img_size": 640, "run_name": "nano_low_iou"},
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.25, "iou_threshold": 0.9,  "img_size": 640, "run_name": "nano_high_iou"},

        # --- Efecto del tamaño de imagen (modelo nano) ---
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.25, "iou_threshold": 0.7,  "img_size": 320, "run_name": "nano_small_img"},
        {"model_path": "yolov8n.pt", "confidence_threshold": 0.25, "iou_threshold": 0.7,  "img_size": 1280, "run_name": "nano_large_img"},
    ]

    for exp in experiments:
        evaluate_model(**exp)