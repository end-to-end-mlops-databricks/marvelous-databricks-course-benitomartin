# Databricks notebook source

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession

from credit_default.utils import load_config

# COMMAND ----------
# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

# Load configuration
config = load_config("project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------

# Load train and test sets
features_balanced = spark.table(f"{catalog_name}.{schema_name}.features_balanced").toPandas()

print(features_balanced)
# COMMAND ----------

# Check for duplicates in the 'Id' column
duplicate_ids = features_balanced[features_balanced["Id"].duplicated()]

if duplicate_ids.empty:
    print("No duplicate IDs found.")
else:
    print(f"Duplicate IDs found:\n{duplicate_ids}")

# COMMAND ----------

# Load source_data
source_data = spark.table(f"{catalog_name}.{schema_name}.source_data").toPandas()

print(source_data)
# COMMAND ----------
source_data.head(103)
# COMMAND ----------

# Check for duplicates in the 'Id' column
duplicate_ids = source_data[source_data["Id"].duplicated()]

if duplicate_ids.empty:
    print("No duplicate IDs found.")
else:
    print(f"Duplicate IDs found:\n{duplicate_ids}")

# COMMAND ----------

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
# COMMAND ----------
train_set.head()
# COMMAND ----------
train_set.tail(334)
# COMMAND ----------
len(train_set)
# COMMAND ----------
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------
len(test_set)
