from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load('titanic_model.joblib')

class Passenger(BaseModel):
    Pclass: int
    Sex: int
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked_Q: int = 0
    Embarked_S: int = 0

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: Passenger):
    df = pd.DataFrame([data.dict()])
    pred = model.predict(df)[^0]
    proba = model.predict_proba(df)[^1]
    
    return {"survived": bool(pred), "survival_probability": proba}
