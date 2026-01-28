import os
import sys
import json
from kafka import KafkaConsumer, KafkaProducer
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml import PipelineModel
from pyspark.ml.feature import StringIndexerModel
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
# MODEL_PATH = "galaxy_rf_pipeline_model"
MODEL_PATH = "full_pipeline_model"
model = PipelineModel.load(MODEL_PATH)
print("Spark model loaded successfully")

# -----------------------------
# Get class names
# -----------------------------
for stage in model.stages:
    if isinstance(stage, StringIndexerModel) and stage.getOutputCol() == "label_index":
        class_names = stage.labels

features = ["u", "g", "r", "i", "z", "u_g", "g_r", "r_i", "i_z"]

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
        input_df = input_df.withColumn("u_g", col("u") - col("g")) \
            .withColumn("g_r", col("g") - col("r")) \
            .withColumn("r_i", col("r") - col("i")) \
            .withColumn("i_z", col("i") - col("z"))

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
