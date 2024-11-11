# Databricks notebook source
import time

import mlflow
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from lightgbm import LGBMRegressor
from mlflow import MlflowClient
from mlflow.models import infer_signature
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import hashlib
import requests

from credit_default.utils import load_config

# COMMAND ----------

# Set up MLflow for tracking and model registry
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

config = load_config("../../project_config.yml")
print(config)

# COMMAND ----------

# Initialize the MLflow client for model management
client = MlflowClient()

# COMMAND ----------

# Extract key configuration details
target = ["Default"]
catalog_name = config.catalog_name
schema_name = config.schema_name
ab_test_params = config.ab_test


# COMMAND ----------



# COMMAND ----------


