from kafka import KafkaProducer
import json
import csv
import time

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

csv_file_path = '../data/creditcard.csv'

with open(csv_file_path, 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # Skip header
    for i, row in enumerate(reader):
        # Create a dictionary from the row
        txn_data = {header[j]: float(row[j]) for j in range(len(header))}
        
        # Send the transaction to Kafka
        producer.send("credit_txn", txn_data)
        print(f"Sent transaction {i+1}")
        
        # Optional: sleep for a short time to simulate a real-time stream
        time.sleep(0.1)

producer.flush()
