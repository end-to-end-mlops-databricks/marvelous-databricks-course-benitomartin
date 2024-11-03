# Databricks notebook source
# dbutils.library.restartPython()

# COMMAND ----------

# # Databricks notebook source

# # COMMAND ----------

# # MAGIC %restart_python

# # MAGIC %pip install --upgrade databricks-sdk

# # COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)

from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from credit_default.utils import load_config

# COMMAND ----------

config = load_config("/Volumes/maven/default/config/project_config.yml")
parameters = config.parameters
print(config)


# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.credit_default_preds"
online_table_name = f"{catalog_name}.{schema_name}.credit_default_preds_online"

# COMMAND ----------

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# COMMAND ----------

df = pd.concat([train_set, test_set])


# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.credit_default_model_basic/3")

# COMMAND ----------

columns = ['Id', 'Limit_bal', 'Sex', 'Education', 'Marriage', 'Age', 'Pay_0',
       'Pay_2', 'Pay_3', 'Pay_4', 'Pay_5', 'Pay_6', 'Bill_amt1', 'Bill_amt2',
       'Bill_amt3', 'Bill_amt4', 'Bill_amt5', 'Bill_amt6', 'Pay_amt1',
       'Pay_amt2', 'Pay_amt3', 'Pay_amt4', 'Pay_amt5', 'Pay_amt6']

# COMMAND ----------

preds_df = df[columns]

# COMMAND ----------

preds_df.loc[:, "Predicted_Default"] = pipeline.predict(df[columns])

# COMMAND ----------

preds_df = spark.createDataFrame(preds_df)

# COMMAND ----------

preds_df

# COMMAND ----------

fe.create_table(
    name=feature_table_name, primary_keys=["Id"], df=preds_df, description="Credit Default predictions feature table"
)

# COMMAND ----------

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table


spec = OnlineTableSpec(
    primary_key_columns=["Id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

# COMMAND ----------

# Create the online table in Databricks
online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)


# COMMAND ----------

columns

# COMMAND ----------

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name, lookup_key="Id", feature_names=[columns]
    )
]

# COMMAND ----------

columns
