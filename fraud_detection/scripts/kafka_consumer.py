from kafka import KafkaConsumer
import json
from inference import predict


consumer = KafkaConsumer(
    'credit_txn',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    auto_offset_reset='earliest',
    group_id='fraud-detector-group'
)

for msg in consumer:
    txn_input = msg.value
    input_array = [txn_input[f"V{i}"] for i in range(1, 31)]
    result = predict(input_array)
    print(f"Transaction result: {result}")