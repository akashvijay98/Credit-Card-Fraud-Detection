import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from model import FraudNet
import joblib

def predict(input_array):
    # Load the saved scaler to normalize input
    scaler = joblib.load("models/scaler.joblib")
    input_scaled = scaler.transform([input_array])  # Keep it 2D

    # Convert to tensor
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Load model
    model = FraudNet(input_dim=30)  # 30 input features
    model.load_state_dict(torch.load("models/best_model.pt", map_location=torch.device('cpu')))
    model.eval()

    # Predict
    with torch.no_grad():
        pred = model(input_tensor)
        probability = float(pred.item())

    return {
        "fraud_probability": probability,
        "prediction": int(probability > 0.5)
    }