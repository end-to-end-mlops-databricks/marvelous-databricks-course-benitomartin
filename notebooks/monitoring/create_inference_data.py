# Databricks notebook source

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from credit_default.utils import load_config

spark = SparkSession.builder.getOrCreate()

# Load configuration
config = load_config("../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
pipeline_id = config.pipeline_id

# Ensure 'Id' column is cast to string in Spark before converting to Pandas
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").withColumn("Id", col("Id").cast("string")).toPandas()

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").withColumn("Id", col("Id").cast("string")).toPandas()


# COMMAND ----------

config = load_config("../../project_config.yml")

print(config)
