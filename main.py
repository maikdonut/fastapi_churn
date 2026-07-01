from fastapi import FastAPI
from schemas import FeatureVectorChurn

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
async def predict(features: FeatureVectorChurn):
    return features