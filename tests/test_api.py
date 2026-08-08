from fastapi.testclient import TestClient
from src.api import app


client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Boston Housing Prediction API is running"


def test_prediction():
    data = {
        "CRIM": 0.00632,
        "ZN": 18,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.09,
        "RAD": 1,
        "TAX": 296,
        "PTRATIO": 15.3,
        "B": 396.9,
        "LSTAT": 4.98
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_MEDV" in result
    assert isinstance(result["predicted_MEDV"], float)


def test_prediction_is_reasonable():
    data = {
        "CRIM": 0.00632,
        "ZN": 18,
        "INDUS": 2.31,
        "CHAS": 0,
        "NOX": 0.538,
        "RM": 6.575,
        "AGE": 65.2,
        "DIS": 4.09,
        "RAD": 1,
        "TAX": 296,
        "PTRATIO": 15.3,
        "B": 396.9,
        "LSTAT": 4.98
    }

    response = client.post("/predict", json=data)

    prediction = response.json()["predicted_MEDV"]

    assert prediction > 0
    assert prediction < 100
