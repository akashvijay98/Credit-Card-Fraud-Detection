import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import joblib
import os

from dataset import FraudDataset
from model import FraudNet

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load preprocessed data
X_train, y_train = joblib.load("data/train.pkl")
X_val, y_val = joblib.load("data/val.pkl")

# Datasets and loaders
train_ds = FraudDataset(X_train, y_train)
val_ds = FraudDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=512)

# Model, loss, optimizer
model = FraudNet(input_dim=X_train.shape[1]).to(device)
criterion = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# Training loop
best_val_loss = float("inf")
for epoch in range(10):
    model.train()
    train_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device).unsqueeze(1)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_val, y_val_ in val_loader:
            x_val = x_val.to(device)
            y_val_ = y_val_.to(device).unsqueeze(1)
            val_preds = model(x_val)
            loss = criterion(val_preds, y_val_)
            val_loss += loss.item()

    print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pt")
        print("Model saved")