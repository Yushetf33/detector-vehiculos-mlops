from fastapi.testclient import TestClient
from src.main import app
import os

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_invalid_file():
    # Enviar un archivo que no es imagen
    response = client.post(
        "/predict",
        files={"file": ("test.txt", b"esto no es una imagen", "text/plain")}
    )
    assert response.status_code == 400

def test_predict_valid_image():
    # test con imagen real pequeña para verificar que devuelve detecciones
    import io
    from PIL import Image
    img = Image.new('RGB', (100, 100), color='white')
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    buf.seek(0)
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    assert "detections" in response.json()
