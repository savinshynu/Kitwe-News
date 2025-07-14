# app/main.py
from fastapi import FastAPI
from schema import TextRequest, PredictionResponse
from model import model_handler

app = FastAPI(title="News Aggregator API")

@app.get("/") # request API point, also known as path operation decorator
def read_root(): # path operation function
    return {"status": "Running great"}

@app.post("/predict", response_model=PredictionResponse)
def get_prediction(request: TextRequest):
    label, confidence = model_handler.predict(request.text)
    return PredictionResponse(label=label, confidence=confidence)
