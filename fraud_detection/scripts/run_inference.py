from inference import predict
import pandas as pd

def main():
    # Read the small csv file
    df = pd.read_csv("fraud_detection/data/creditcard_small.csv")

    # Drop the 'Class' column if it exists
    if 'Class' in df.columns:
        df = df.drop('Class', axis=1)

    # Iterate over each row and make a prediction
    for index, row in df.iterrows():
        features = row.tolist()
        result = predict(features)
        prediction = "Fraud" if result["prediction"] == 1 else "Not Fraud"
        print(f"Row {index}: {prediction} (Probability: {result['fraud_probability']:.4f})")

if __name__ == "__main__":
    main()
