import os

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

from app.preprocessing import prepare_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "extra_trees_phase5_light.joblib"
)

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="V2X Driver Behaviour Detection API",
    version="1.0"
)


class V2XInput(BaseModel):
    PC_1: float
    PC_2: float
    PC_3: float
    PC_4: float
    PC_5: float
    PC_6: float
    PC_7: float
    PC_8: float
    PC_9: float
    PC_10: float
    PC_11: float
    PC_12: float
    PC_13: float
    PC_14: float
    PC_15: float
    PC_16: float
    PC_17: float
    PC_18: float


@app.get("/")
def home():
    return {
        "status": "running",
        "application": "V2X Driver Behaviour Detection"
    }


@app.post("/predict")
def predict(data: V2XInput):

    record = data.model_dump()

    X = prepare_features(record)

    prediction = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        probability = float(
            model.predict_proba(X)[0][1]
        )
    else:
        probability = float(prediction)

    label = (
        "Attacker"
        if prediction == 1
        else "Normal"
    )

    return {
        "prediction": prediction,
        "label": label,
        "attacker_probability": round(
            probability, 4
        )
    }
