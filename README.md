# Proyecto MLOps - Detección de Vehículos

API REST para detección de vehículos en imágenes usando YOLOv8 y FastAPI, contenerizada con Docker y con seguimiento de experimentos mediante Weights & Biases.

## Autor
**Yushe** — Máster en Deep Learning, Universidad Politécnica de Madrid

## Instalación

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Ejecutar la API

```bash
uvicorn src.main:app --reload
```

Accede a la documentación en: http://localhost:8000/docs

## Endpoints

- `GET /` — Estado de la API
- `GET /health` — Health check
- `POST /predict` — Detecta vehículos en una imagen (enviar como form-data)

## Tests

```bash
pytest tests/
```

## Docker

```bash
docker build -t vehicle-detector .
docker run -p 8000:8000 vehicle-detector
```

## Experimentos W&B

Los experimentos y métricas del modelo están disponibles en el siguiente reporte de Weights & Biases:
[Enlace al reporte W&B]

## Estructura del proyecto

```
├── src/
│   ├── main.py        # API FastAPI
│   ├── predict.py     # Lógica de detección con YOLOv8
│   └── train.py       # Evaluación y registro de experimentos en W&B
├── tests/
│   └── test_api.py    # Tests de la API
├── data/              # Dataset
├── models/            # Modelos entrenados
├── notebooks/         # Notebooks de exploración
├── Dockerfile
└── requirements.txt
```
