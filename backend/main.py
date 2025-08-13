# app/main.py
from fastapi import FastAPI
from schema import TextRequest, PredictionResponse
from model_quantized import model_handler

app = FastAPI(title="News Aggregator API")

@app.get("/") # request API point, also known as path operation decorator
async def read_root(): # path operation function
    return {"status": "Running great"}

@app.post("/predict", response_model=PredictionResponse)
async def get_prediction(news: TextRequest):
    label, confidence = model_handler.predict(news.texts)
    return PredictionResponse(label=label, confidence=confidence)
