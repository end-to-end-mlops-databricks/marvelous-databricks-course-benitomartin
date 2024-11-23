# Databricks notebook source
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient

# from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructField, StructType

from credit_default.utils import load_config

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

workspace = WorkspaceClient()

# Load configuration
config = load_config("../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
target = config.target[0].new_name

inf_table = spark.sql(f"SELECT * FROM {catalog_name}.{schema_name}.`model-serving-fe_payload`")

# COMMAND ----------

## Dataframe records on payload table under response column
# {"dataframe_records": [{"Id": "43565", "Limit_bal": 198341.0, "Sex": 2.0,
# "Education": 2.0, "Marriage": 2.0, "Age": 26.0, "Pay_0": 2.0, "Pay_2": 1.0,
# "Pay_3": 6.0, "Pay_4": 4.0, "Pay_5": 8.0, "Pay_6": 6.0, "Bill_amt1": -44077.0,
# "Bill_amt2": 15797.0, "Bill_amt3": 66567.0, "Bill_amt4": 54582.0, "Bill_amt5": 79211.0,
# "Bill_amt6": 129060.0, "Pay_amt1": 13545.0, "Pay_amt2": 20476.0, "Pay_amt3": 8616.0,
# "Pay_amt4": 3590.0, "Pay_amt5": 22999.0, "Pay_amt6": 3605.0}]}
request_schema = StructType(
    [
        StructField(
            "dataframe_records",
            ArrayType(
                StructType(
                    [
                        StructField("Id", StringType(), True),
                        StructField("Limit_bal", DoubleType(), True),
                        StructField("Sex", DoubleType(), True),
                        StructField("Education", DoubleType(), True),
                        StructField("Marriage", DoubleType(), True),
                        StructField("Age", DoubleType(), True),
                        StructField("Pay_0", DoubleType(), True),
                        StructField("Pay_2", DoubleType(), True),
                        StructField("Pay_3", DoubleType(), True),
                        StructField("Pay_4", DoubleType(), True),
                        StructField("Pay_5", DoubleType(), True),
                        StructField("Pay_6", DoubleType(), True),
                        StructField("Bill_amt1", DoubleType(), True),
                        StructField("Bill_amt2", DoubleType(), True),
                        StructField("Bill_amt3", DoubleType(), True),
                        StructField("Bill_amt4", DoubleType(), True),
                        StructField("Bill_amt5", DoubleType(), True),
                        StructField("Bill_amt6", DoubleType(), True),
                        StructField("Pay_amt1", DoubleType(), True),
                        StructField("Pay_amt2", DoubleType(), True),
                        StructField("Pay_amt3", DoubleType(), True),
                        StructField("Pay_amt4", DoubleType(), True),
                        StructField("Pay_amt5", DoubleType(), True),
                        StructField("Pay_amt6", DoubleType(), True),
                    ]
                )
            ),
            True,
        )
    ]
)

# Standard Databricks schema for the response
response_schema = StructType(
    [
        StructField("predictions", ArrayType(DoubleType()), True),
        StructField(
            "databricks_output",
            StructType(
                [StructField("trace", StringType(), True), StructField("databricks_request_id", StringType(), True)]
            ),
            True,
        ),
    ]
)
# COMMAND ----------

# Parse the request and response columns in one Dataframe
inf_table_parsed = inf_table.withColumn("parsed_request", F.from_json(F.col("request"), request_schema))

inf_table_parsed = inf_table_parsed.withColumn("parsed_response", F.from_json(F.col("response"), response_schema))

df_exploded = inf_table_parsed.withColumn("record", F.explode(F.col("parsed_request.dataframe_records")))


df_final = df_exploded.select(
    F.from_unixtime(F.col("timestamp_ms") / 1000).cast("timestamp").alias("timestamp"),
    "timestamp_ms",
    "databricks_request_id",
    "execution_time_ms",
    F.col("record.Id").alias("Id"),
    F.col("record.Limit_bal").alias("Limit_bal"),
    F.col("record.Sex").alias("Sex"),
    F.col("record.Education").alias("Education"),
    F.col("record.Marriage").alias("Marriage"),
    F.col("record.Age").alias("Age"),
    F.col("record.Pay_0").alias("Pay_0"),
    F.col("record.Pay_2").alias("Pay_2"),
    F.col("record.Pay_3").alias("Pay_3"),
    F.col("record.Pay_4").alias("Pay_4"),
    F.col("record.Pay_5").alias("Pay_5"),
    F.col("record.Pay_6").alias("Pay_6"),
    F.col("record.Bill_amt1").alias("Bill_amt1"),
    F.col("record.Bill_amt2").alias("Bill_amt2"),
    F.col("record.Bill_amt3").alias("Bill_amt3"),
    F.col("record.Bill_amt4").alias("Bill_amt4"),
    F.col("record.Bill_amt5").alias("Bill_amt5"),
    F.col("record.Bill_amt6").alias("Bill_amt6"),
    F.col("record.Pay_amt1").alias("Pay_amt1"),
    F.col("record.Pay_amt2").alias("Pay_amt2"),
    F.col("record.Pay_amt3").alias("Pay_amt3"),
    F.col("record.Pay_amt4").alias("Pay_amt4"),
    F.col("record.Pay_amt5").alias("Pay_amt5"),
    F.col("record.Pay_amt6").alias("Pay_amt6"),
    F.col("parsed_response.predictions")[0].alias("prediction"),
    F.lit("credit_model_feature").alias("model_name"),
)
# COMMAND ----------

# Join train/test/inference sets with request/response data
test_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
inference_set_normal = spark.table(f"{catalog_name}.{schema_name}.inference_set_normal")
inference_set_skewed = spark.table(f"{catalog_name}.{schema_name}.inference_set_skewed")

inference_set = inference_set_normal.union(inference_set_skewed)


df_final_with_status = (
    df_final.join(test_set.select("Id", target), on="Id", how="left")
    .withColumnRenamed(target, "default_test")
    .join(inference_set.select("Id", target), on="Id", how="left")
    .withColumnRenamed(target, "default_inference")
    .select("*", F.coalesce(F.col("default_test"), F.col("default_inference")).alias("default"))
    .drop("default_test", "default_inference")
    .withColumn("default", F.col("default").cast("double"))
    .withColumn("prediction", F.col("prediction").cast("double"))
    .dropna(subset=["default", "prediction"])
)

# COMMAND ----------
df_final_with_status = df_final_with_status.dropDuplicates(["Id"])

# COMMAND ----------
df_final_with_status.groupBy("Id").count().filter(F.col("count") > 1).show()


# COMMAND ----------
features_balanced = spark.table(f"{catalog_name}.{schema_name}.features_balanced")

# COMMAND ----------
features_balanced = df_final_with_status.dropDuplicates(["Id"])
df_final_with_status.groupBy("Id").count().filter(F.col("count") > 1).show()


# COMMAND ----------
df_final_with_features = df_final_with_status.join(features_balanced, on="Id", how="left")


# COMMAND ----------
df_final_with_features.write.format("delta").mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.model_monitoring"
)


# COMMAND ----------
df_final_with_features.printSchema()

# COMMAND ----------
features_balanced.printSchema()

# COMMAND ----------
df_joined = df_final_with_status.join(features_balanced, on="Id", how="left")
df_joined.printSchema()

# COMMAND ----------
# Check duplicates in df_final_with_status
df_final_with_status.groupBy("Id").count().filter(F.col("count") > 1).show()

# COMMAND ----------
# Check duplicates in features_balanced
features_balanced.groupBy("Id").count().filter(F.col("count") > 1).show()

# COMMAND ----------
df_final_with_status = df_final_with_status.dropDuplicates(["Id"])

# COMMAND ----------
df_final_with_features.write.format("delta").mode("overwrite").saveAsTable(
    f"{catalog_name}.{schema_name}.model_monitoring"
)


# # COMMAND ----------
# workspace.quality_monitors.run_refresh(
#     table_name=f"{catalog_name}.{schema_name}.model_monitoring"
# )
# COMMAND ----------
