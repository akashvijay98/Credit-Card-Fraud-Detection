from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

sample_txn = {
    "V1": -1.3598071, "V2": -0.0727, "V30": 0.1234  # 30 values
}

producer.send("credit_txn", sample_txn)
producer.flush()
