# Proyecto MLOps - Detección de Vehículos

API REST para detección de vehículos en imágenes usando YOLOv8 y FastAPI.

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

## Tests

```bash
pytest tests/
```

## Docker

```bash
docker build -t vehicle-detector .
docker run -p 8000:8000 vehicle-detector
```