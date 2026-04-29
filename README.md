# 🚗 Proyecto MLOps - Detección de Vehículos

## Sobre el proyecto
Este proyecto surge del trabajo realizado en la asignatura de Visión por Computador 
del Máster, donde entrené modelos YOLOv8 para detección de objetos. Quise ir un paso 
más allá y operacionalizar el modelo: convertirlo en un servicio real accesible desde 
cualquier lugar.

API REST para detección de vehículos en imágenes usando YOLOv8 y FastAPI, contenerizada con Docker y con seguimiento de experimentos mediante Weights & Biases.

## Autor
**Yushetf**

## Descripción
Este proyecto aplica metodologías MLOps a un sistema de detección de vehículos basado en YOLOv8. El modelo detecta coches, camiones, autobuses y motocicletas en imágenes, devolviendo las detecciones con clase, confianza y bounding box. Se han realizado 9 experimentos comparando distintos modelos y configuraciones, registrados en Weights & Biases.

## 🛠️ Tecnologías utilizadas
- **YOLOv8** — Modelo de detección de objetos
- **FastAPI** — Framework para la API REST
- **Docker** — Contenedorización del servicio
- **Weights & Biases** — Tracking de experimentos
- **pytest** — Tests automatizados

## 📁 Estructura del proyecto

```text
.
├── src/
│   ├── main.py          # API FastAPI con endpoints de detección
│   ├── predict.py       # Lógica de detección con YOLOv8
│   └── train.py         # Evaluación y registro de experimentos en W&B
├── tests/
│   └── test_api.py      # Tests automatizados de la API
├── notebooks/
│   └── ProyectoDL_Vision.ipynb  # Notebook del proyecto original
├── data/                # Dataset
├── models/              # Modelos entrenados
├── Dockerfile
├── requirements.txt
└── README.md

## ⚙️ Instalación local

```bash
# Clonar el repositorio
git clone https://github.com/Yushetf33/detector-vehiculos-mlops.git
cd detector-vehiculos-mlops

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

## 🚀 Ejecutar la API

```bash
uvicorn src.main:app --reload
```

Accede a la documentación interactiva en: http://localhost:8000/docs

## 📡 Endpoints

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Estado de la API |
| GET | `/health` | Health check |
| POST | `/predict` | Detecta vehículos en una imagen |

### Ejemplo de respuesta `/predict`
```json
{
  "filename": "imagen.jpg",
  "num_detections": 3,
  "detections": [
    {"class": "car", "confidence": 0.87, "bbox": [120, 45, 380, 210]},
    {"class": "truck", "confidence": 0.76, "bbox": [400, 60, 650, 280]}
  ]
}
```

## 🧪 Tests

```bash
pytest tests/
```

## 🐳 Docker

```bash
# Construir imagen
docker build -t vehicle-detector .

# Ejecutar contenedor
docker run -p 8000:8000 vehicle-detector
```

## 📊 Experimentos W&B

Se han realizado 9 experimentos comparando distintas configuraciones:
- Comparativa de modelos: YOLOv8n, YOLOv8s, YOLOv8m
- Impacto del umbral de confianza (0.10, 0.25, 0.50)
- Impacto del umbral IoU (0.5, 0.7, 0.9)
- Impacto de la resolución de entrada (320, 640, 1280)

🔗 [Ver reporte completo en W&B](https://api.wandb.ai/links/yushetf-universidad-polit-cnica-de-madrid/ftqfr6sb)

## 🌐 Servicio en producción

API desplegada y accesible en:
👉 **https://detector-vehiculos-mlops.onrender.com**

Documentación interactiva:
👉 **https://detector-vehiculos-mlops.onrender.com/docs**
