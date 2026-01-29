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
MODEL_PATH_1 = "full_pipeline_model"
MODEL_PATH_2 = "galaxy_rf_pipeline_model"

model_1 = PipelineModel.load(MODEL_PATH_1)
print("Spark model_1 loaded successfully")

model_2 = PipelineModel.load(MODEL_PATH_2)
print("Spark model_2 loaded successfully")

# -----------------------------
# Get class names
# -----------------------------
for stage in model_1.stages:
    if isinstance(stage, StringIndexerModel) and stage.getOutputCol() == "label_index":
        class_names_1 = stage.labels

class_names_2 = ['GALAXY', 'STAR', 'QSO']
# -----------------------------
# Continuous listening loop
# -----------------------------
print("Consumer waiting for data... (press Ctrl+C to stop)")

try:
    for message in consumer:
        data = message.value
        print(f"Received input: {data}")
        model_n = data[0]
        data = data[1]

        # Convert dict to Spark DataFrame
        input_df = spark.createDataFrame([Row(**data)])
        if model_n == 'm1':
            model = model_1
            class_names = class_names_1
            input_df = input_df.withColumn("u_g", col("u") - col("g")) \
                .withColumn("g_r", col("g") - col("r")) \
                .withColumn("r_i", col("r") - col("i")) \
                .withColumn("i_z", col("i") - col("z"))
        else:
            model = model_2
            class_names = class_names_2

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
