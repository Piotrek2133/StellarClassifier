import os
import sys
import json
from kafka import KafkaConsumer, KafkaProducer
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql import Row

# -----------------------------
# Force PySpark to use the same Python
# -----------------------------
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# -----------------------------
# Kafka setup
# -----------------------------
DATA_TOPIC = "galaxy-input"
PRED_TOPIC = "galaxy-prediction"

consumer = KafkaConsumer(
    DATA_TOPIC,
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    group_id="consumer-group",
    value_deserializer=lambda v: json.loads(v.decode("utf-8"))
)

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# -----------------------------
# Load Spark model
# -----------------------------
spark = SparkSession.builder.appName(
    "GalaxyClassification-Consumer").getOrCreate()
MODEL_PATH = "galaxy_rf_pipeline_model"
model = PipelineModel.load(MODEL_PATH)
print("Spark model loaded successfully")

# -----------------------------
# Get feature columns from training data
# -----------------------------
train_df = spark.read.csv("clean.csv", header=True, inferSchema=True)
feature_cols = [c for c in train_df.columns if c != "class"]

# Map numeric label to class name
class_names = (
    train_df.select("class")
    .distinct()
    .orderBy("class")
    .rdd.flatMap(lambda x: x)
    .collect()
)

# -----------------------------
# Continuous listening loop
# -----------------------------
print("Consumer waiting for data... (press Ctrl+C to stop)")

try:
    for message in consumer:
        data = message.value
        print(f"Received input: {data}")

        # Convert dict to Spark DataFrame
        input_df = spark.createDataFrame([Row(**data)])

        # Predict
        predictions = model.transform(input_df)
        pred_label = int(predictions.select("prediction").first()[0])
        predicted_class = class_names[pred_label]

        # Send back prediction
        producer.send(PRED_TOPIC, value=predicted_class)
        producer.flush()
        print(f"Predicted class sent back: {predicted_class}\n")

except KeyboardInterrupt:
    print("Consumer stopped by user.")

finally:
    consumer.close()
    producer.close()
    spark.stop()
