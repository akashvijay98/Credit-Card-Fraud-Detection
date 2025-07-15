from inference import predict

def main():
    # Example input: 30 features (replace with real transaction data as needed)
    sample_input = [0.1] * 30

    # Call the predict function
    result = predict(sample_input)

    # Print the output
    print("Prediction output:", result)

if __name__ == "__main__":
    main()
