import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import joblib
import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

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
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x_val, y_val_ in val_loader:
            x_val = x_val.to(device)
            y_val_ = y_val_.to(device).unsqueeze(1)
            val_preds = model(x_val)
            loss = criterion(val_preds, y_val_)
            val_loss += loss.item()
            all_preds.append(val_preds.cpu().numpy())
            all_labels.append(y_val_.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Convert probabilities to binary predictions
    binary_preds = (all_preds > 0.5).astype(int)

    precision = precision_score(all_labels, binary_preds, zero_division=0)
    recall = recall_score(all_labels, binary_preds, zero_division=0)
    f1 = f1_score(all_labels, binary_preds, zero_division=0)

    print(
        f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 = {f1:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "models/best_model.pt")
        print("Model saved")