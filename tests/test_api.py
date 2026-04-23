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