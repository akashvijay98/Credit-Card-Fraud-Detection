import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from model_handler import predict_transaction

app = FastAPI()

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(tx: Transaction):
    return predict_transaction(tx.features)