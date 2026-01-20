from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
import sys
import os

# Force PySpark to use the same Python as the current interpreter
python_bin = sys.executable
os.environ["PYSPARK_PYTHON"] = python_bin
os.environ["PYSPARK_DRIVER_PYTHON"] = python_bin


# ----------------------------
# Spark session
# ----------------------------
spark = SparkSession.builder \
    .appName("GalaxyClassification-LoadModel") \
    .getOrCreate()

# ----------------------------
# Load trained pipeline model
# ----------------------------
MODEL_PATH = "galaxy_rf_pipeline_model"
model = PipelineModel.load(MODEL_PATH)

print("Model loaded successfully")

# ----------------------------
# Load training data schema
# (to know exact feature columns)
# ----------------------------
train_df = spark.read.csv(
    "clean.csv",
    header=True,
    inferSchema=True
)

label_col = "class"
feature_cols = [c for c in train_df.columns if c != label_col]

print("Expected feature columns:")
for c in feature_cols:
    print(" -", c)

# ----------------------------
# Example input data
# IMPORTANT: replace values with real ones
# ----------------------------
input_data = [{
    col: float(train_df.select(col).first()[0]) if train_df.select(
        col).first()[0] is not None else 0.0
    for col in feature_cols
}]

input_df = spark.createDataFrame(input_data)

print("Input data:")
input_df.show(truncate=False)

# ----------------------------
# Run prediction
# ----------------------------
predictions = model.transform(input_df)

predictions.select(
    "prediction",
    "probability"
).show(truncate=False)

# ----------------------------
# Optional: map numeric label to class name
# ----------------------------
class_names = (
    train_df.select("class")
    .distinct()
    .orderBy("class")
    .rdd.flatMap(lambda x: x)
    .collect()
)

pred_label = int(predictions.select("prediction").first()[0])
print(f"Predicted class: {class_names[pred_label]}")

spark.stop()
