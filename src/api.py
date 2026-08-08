import os
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

MODEL_PATH = "models/best_model.pkl"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Boston Housing Prediction API",
    description="API for predicting Boston housing prices",
    version="1.0.0"
)


# --------------------------------------------------
# Input schema
# --------------------------------------------------

class HousingData(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Boston Housing Prediction API is running"
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: HousingData):

    input_data = pd.DataFrame([{
        "CRIM": data.CRIM,
        "ZN": data.ZN,
        "INDUS": data.INDUS,
        "CHAS": data.CHAS,
        "NOX": data.NOX,
        "RM": data.RM,
        "AGE": data.AGE,
        "DIS": data.DIS,
        "RAD": data.RAD,
        "TAX": data.TAX,
        "PTRATIO": data.PTRATIO,
        "B": data.B,
        "LSTAT": data.LSTAT
    }])

    prediction = model.predict(input_data)[0]

    return {
        "predicted_MEDV": round(float(prediction), 2)
    }
