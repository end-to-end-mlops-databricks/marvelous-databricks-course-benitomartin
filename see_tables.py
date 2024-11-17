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

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
# COMMAND ----------
train_set.head()
# COMMAND ----------
train_set.tail()
# COMMAND ----------
len(train_set)
# COMMAND ----------
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------
len(test_set)
# COMMAND ----------
feature = spark.table(f"{catalog_name}.{schema_name}.features_balanced").toPandas()
# COMMAND ----------
len(feature)

# COMMAND ----------
# Check if values in the 'Id' column start with 'Id_'
train_set["StartsWith_Id_"] = train_set["Id"].str.startswith("id_")

print(train_set)
# COMMAND ----------
# COMMAND ----------
# Check if values in the 'Id' column start with 'Id_'
feature["StartsWith_Id_"] = feature["Id"].str.startswith("id_")

print(feature)
# COMMAND ----------
# Count rows where 'Id' starts with 'Id_'
count = feature["Id"].str.startswith("id_").sum()

print(f"Number of rows where 'Id' starts with 'id_': {count}")
# COMMAND ----------
train_set[["Id"]] = train_set[["Id"]].astype(int)
# COMMAND ----------
feature.tail()
# COMMAND ----------
train_set.info()
# COMMAND ----------
train_set.Id.max()
# COMMAND ----------
len(train_set)
# COMMAND ----------
