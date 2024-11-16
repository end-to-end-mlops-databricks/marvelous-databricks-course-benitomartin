"""
This script handles data ingestion and feature table updates for a house price prediction system.

Key functionality:
- Loads the source dataset and identifies new records for processing
- Splits new records into train and test sets based on timestamp
- Updates existing train and test tables with new data
- Inserts the latest feature values into the feature table for serving
- Triggers and monitors pipeline updates for online feature refresh
- Sets task values to coordinate pipeline orchestration

Workflow:
1. Load source dataset and retrieve recent records with updated timestamps.
2. Split new records into train and test sets (80-20 split).
3. Append new train and test records to existing train and test tables.
4. Insert the latest feature data into the feature table for online serving.
5. Trigger a pipeline update and monitor its status until completion.
6. Set a task value indicating whether new data was processed.
"""

import argparse

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession

from credit_default.utils import load_config

workspace = WorkspaceClient()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"
config = load_config(config_path)
pipeline_id = config.pipeline_id

spark = SparkSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name
