from fastapi import FastAPI
from schemas import FeatureVectorChurn, DatasetRowChurn
from dataset import dataset_service


app = FastAPI()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}

@app.post("/predict")
def predict(features: FeatureVectorChurn):
    return features

@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def preview(n: int = 5):
    return dataset_service.preview(n)

@app.get("/dataset/info")
def info():
    return dataset_service.info()