import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def prepare_and_save_data():
    print("Starting data preparation...")

    # Check if file exists
    csv_path = "data/creditcard.csv"
    if not os.path.exists(csv_path):
        print(f"ERROR: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully! Shape: {df.shape}")

    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features normalized")

    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    print("Scaler saved to models/scaler.joblib")

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train and validation split done. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    os.makedirs("data", exist_ok=True)
    joblib.dump((X_train, y_train), "data/train.pkl")
    joblib.dump((X_val, y_val), "data/val.pkl")
    print("Train and validation data saved to data/train.pkl and data/val.pkl")

if __name__ == "__main__":
    prepare_and_save_data()
