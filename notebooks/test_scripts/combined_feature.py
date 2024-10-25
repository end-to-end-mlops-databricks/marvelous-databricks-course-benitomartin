# Databricks notebook source
import os

import mlflow
import pandas as pd
from databricks import feature_engineering
from databricks.feature_store import FeatureLookup
from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from pyspark.sql import SparkSession

from credit_default.utils import load_config

spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()
# Load environment variables
load_dotenv()

CONFIG_DATABRICKS = os.environ["CONFIG_DATABRICKS"]
PROFILE = os.environ["PROFILE"]
print(CONFIG_DATABRICKS)

# COMMAND ----------


# COMMAND ----------
# Load configuration from YAML file
config = load_config(CONFIG_DATABRICKS)
catalog_name = config["catalog_name"]
schema_name = config["schema_name"]
parameters = config["parameters"]

# COMMAND ----------

test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").drop("Update_timestamp_utc").toPandas()
# COMMAND ----------

train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("Update_timestamp_utc")

# COMMAND ----------

train_set.take(5)

# COMMAND ----------

# tracking and registry URIs
mlflow.set_tracking_uri(f"databricks://{PROFILE}")
mlflow.set_registry_uri(f"databricks-uc://{PROFILE}")
workspace = WorkspaceClient()


def compute_features(df):
    pdf = df.toPandas()
    X = pdf.drop(columns=["Default"])
    y = pdf["Default"]

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_df = pd.DataFrame(X_resampled)
    resampled_df["Default"] = y_resampled

    return spark.createDataFrame(resampled_df)


train_df_smote = compute_features(train_set)


columns = [
    "Limit_bal",
    "Sex",
    "Education",
    "Marriage",
    "Age",
    "Pay_0",
    "Pay_2",
    "Pay_3",
    "Pay_4",
    "Pay_5",
    "Pay_6",
    "Bill_amt1",
    "Bill_amt2",
    "Bill_amt3",
    "Bill_amt4",
    "Bill_amt5",
    "Bill_amt6",
    "Pay_amt1",
    "Pay_amt2",
    "Pay_amt3",
    "Pay_amt4",
    "Pay_amt5",
    "Pay_amt6",
    "Default",
]

training_set = fe.create_training_set(
    df=train_df_smote,
    label="Default",
    feature_lookups=[
        FeatureLookup(
            table_name=f"{catalog_name}.{schema_name}.train_set_smote",
            feature_names=columns,
            lookup_key="Sex",
        )
    ],
)
