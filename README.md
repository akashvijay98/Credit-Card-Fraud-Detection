# Credit Card Fraud Detection

## Project Overview

This project implements a machine learning solution for detecting fraudulent credit card transactions. It includes scripts for data preparation, model training, and real-time inference. The core of the project is a deep learning model built with PyTorch, served via a FastAPI application. It also includes a streaming inference pipeline using Apache Kafka.

## Features

- **Deep Learning Model**: A neural network built with PyTorch for fraud classification.
- **RESTful API**: A FastAPI endpoint for on-demand, real-time predictions.
- **Streaming Pipeline**: Real-time fraud detection using Kafka for message passing.
- **Scalable and Modular**: The project is structured to be easily extensible.

## System Requirements

- Python 3.8+
- PyTorch
- FastAPI
- scikit-learn
- pandas
- joblib
- Kafka (for streaming)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd credit-card-fraud-detection
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: A `requirements.txt` file is not provided. You will need to create one based on the imports in the project files.)*

3.  **Download the dataset:**
    - Download the "Credit Card Fraud Detection" dataset from Kaggle.
    - Place the `creditcard.csv` file in the `data/` directory.

## Usage

### 1. Data Preparation

First, you need to preprocess the raw data. This script will scale the features and create training and validation sets.

```bash
python scripts/prepare_data.py
```

This will create `train.pkl`, `val.pkl` in the `data/` directory and `scaler.joblib` in the `models/` directory.

### 2. Model Training

Next, train the neural network model:

```bash
python scripts/train.py
```

This script will train the model and save the best-performing version as `best_model.pt` in the `models/` directory.

### 3. Running the API

To serve the model via a REST API, run the following command:

```bash
uvicorn api.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 4. Real-time Inference with Kafka

For streaming predictions, you need to have Kafka running.

1.  **Start the Kafka consumer:**
    This script will listen for incoming transactions and run predictions.
    ```bash
    python scripts/kafka_consumer.py
    ```

2.  **Start the Kafka producer:**
    This script will simulate a stream of transactions by sending data from the CSV file to the Kafka topic.
    ```bash
    python scripts/kafka_producer.py
    ```

Predicted results will be printed by the consumer and also sent to the `credit_txn_results` Kafka topic.

## API Endpoint

### POST /predict

-   **URL**: `/predict`
-   **Method**: `POST`
-   **Body**:
    ```json
    {
      "features": [
        0.0, -1.35, ..., 0.73, 0.0, 1.23, -0.68, -0.23, -0.56, 0.0, 0.0
      ]
    }
    ```
-   **Response**:
    ```json
    {
      "fraud_probability": 0.99,
      "prediction": 1
    }
    ```

## Project Structure

```
.
├── api/
│   ├── main.py
│   └── model_handler.py
├── data/
│   ├── creditcard.csv
│   └── ...
├── models/
│   ├── best_model.pt
│   └── scaler.joblib
├── scripts/
│   ├── dataset.py
│   ├── inference.py
│   ├── kafka_consumer.py
│   ├── kafka_producer.py
│   ├── model.py
│   ├── prepare_data.py
│   ├── run_inference.py
│   └── train.py
└── README.md
```
