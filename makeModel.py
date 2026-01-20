from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, rand

from pyspark.ml.feature import (
    StringIndexer,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml.classification import RandomForestClassifier

# ------------------------
# Spark session
# ------------------------
spark = SparkSession.builder \
    .appName("GalaxyClassification-Train") \
    .getOrCreate()

# ------------------------
# Load data
# ------------------------
df = spark.read.csv(
    "clean.csv",
    header=True,
    inferSchema=True
)

label_col = "class"
feature_cols = [c for c in df.columns if c != label_col]

# ------------------------
# Label encoding
# ------------------------
label_indexer = StringIndexer(
    inputCol="class",
    outputCol="label"
)

df = label_indexer.fit(df).transform(df)

# ------------------------
# Balance classes
# ------------------------
class_counts = df.groupBy("label").count().collect()
max_count = max(row["count"] for row in class_counts)

balanced_dfs = []

for row in class_counts:
    label = row["label"]
    count = row["count"]
    ratio = max_count / count

    sampled = (
        df.filter(col("label") == label)
          .sample(withReplacement=True, fraction=ratio, seed=42)
    )
    balanced_dfs.append(sampled)

df_balanced = balanced_dfs[0]
for temp_df in balanced_dfs[1:]:
    df_balanced = df_balanced.union(temp_df)

# ------------------------
# Pipeline stages
# ------------------------
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True
)

rf = RandomForestClassifier(
    featuresCol="scaledFeatures",
    labelCol="label",
    seed=42
)

pipeline = Pipeline(stages=[
    assembler,
    scaler,
    rf
])

# ------------------------
# Train
# ------------------------
train_df, test_df = df_balanced.randomSplit([0.67, 0.33], seed=42)
pipeline_model = pipeline.fit(train_df)

# ------------------------
# Save model
# ------------------------
MODEL_PATH = "galaxy_rf_pipeline_model"

pipeline_model.write().overwrite().save(MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")

spark.stop()
