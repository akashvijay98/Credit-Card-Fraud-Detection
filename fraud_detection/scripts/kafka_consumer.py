from kafka import KafkaConsumer, KafkaProducer
import json
from inference import predict

# Consumer to read from the input topic
consumer = KafkaConsumer(
    'credit_txn',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='fraud-detector-group'
)

# Producer to write to the results topic
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

for msg in consumer:
    txn_input = msg.value
    input_array = [txn_input[f"V{i}"] for i in range(1, 31)]
    result = predict(input_array)
    
    # Combine input transaction with the prediction result
    output_data = {
        "transaction": txn_input,
        "prediction": result
    }
    
    # Send the result to the new topic
    producer.send("credit_txn_results", output_data)
    producer.flush()
    
    print(f"Processed transaction and sent result: {output_data}")
