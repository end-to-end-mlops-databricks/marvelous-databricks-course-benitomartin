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
import time

# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession
from databricks.sdk import WorkspaceClient
from loguru import logger
from pyspark.sql.functions import col
from pyspark.sql.functions import max as spark_max

from credit_default.utils import load_config

workspace = WorkspaceClient()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default="",
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
config_path = f"{root_path}/project_config.yml"
config = load_config(config_path)
pipeline_id = config.pipeline_id

# spark = SparkSession.builder.getOrCreate()
spark = DatabricksSession.builder.getOrCreate()

catalog_name = config.catalog_name
schema_name = config.schema_name


# Load source_data table
source_data_table_name = f"{catalog_name}.{schema_name}.source_data"
source_data = spark.table(f"{source_data_table_name}")

# Get max update timestamps from existing data
max_train_timestamp = (
    spark.table(f"{catalog_name}.{schema_name}.train_set")
    .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

max_test_timestamp = (
    spark.table(f"{catalog_name}.{schema_name}.test_set")
    .select(spark_max("Update_timestamp_utc").alias("max_update_timestamp"))
    .collect()[0]["max_update_timestamp"]
)

latest_timestamp = max(max_train_timestamp, max_test_timestamp)

# Filter source_data for rows with update_timestamp_utc greater than the latest_timestamp
new_data = source_data.filter(col("Update_timestamp_utc") > latest_timestamp)

logger.info(f"Loading {new_data.count()} data from {source_data_table_name}")


# Split the new data into train and test sets
new_data_train, new_data_test = new_data.randomSplit([0.8, 0.2], seed=42)

# Update train_set and test_set tables
new_data_train.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.train_set")
new_data_test.write.mode("append").saveAsTable(f"{catalog_name}.{schema_name}.test_set")

# Verify affected rows count for train and test
affected_rows_train = new_data_train.count()
affected_rows_test = new_data_test.count()

logger.info(f"New train data {affected_rows_train}")
logger.info(f"New test data {affected_rows_test}")

columns = config.features.clean

# Write into feature table; update online table
if affected_rows_train > 0 or affected_rows_test > 0:
    # Construct the column selection part of the query
    columns_str = ", ".join(f"s.{col}" for col in columns)

    # Get all new records that need to be added to feature table
    spark.sql(f"""
        INSERT INTO {catalog_name}.{schema_name}.features_balanced
        SELECT DISTINCT {columns_str}
        FROM {catalog_name}.{schema_name}.source_data s
        JOIN (
            SELECT Id, Update_timestamp_utc
            FROM {catalog_name}.{schema_name}.train_set
            WHERE Update_timestamp_utc > '{latest_timestamp}'
            UNION ALL
            SELECT Id, Update_timestamp_utc
            FROM {catalog_name}.{schema_name}.test_set
            WHERE Update_timestamp_utc > '{latest_timestamp}'
        ) new_records
        ON s.Id = new_records.Id
        WHERE s.Update_timestamp_utc > '{latest_timestamp}'
    """)

    refreshed = 1

    # This updates the "online" feature table with the new data from the "offline" feature table.
    # full_refresh=False means the pipeline will incrementally update the data.
    # This allows the pipeline to update only the new or changed records,
    # which can be much more efficient than a full refresh, especially for large datasets.
    update_response = workspace.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)

    while True:
        update_info = workspace.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_response.update_id)
        state = update_info.update.state.value
        if state == "COMPLETED":
            break
        elif state in ["FAILED", "CANCELED"]:
            raise SystemError("Online table failed to update.")
        elif state == "WAITING_FOR_RESOURCES":
            print("Pipeline is waiting for resources.")
        else:
            print(f"Pipeline is in {state} state.")
        time.sleep(30)

else:
    refreshed = 0

dbutils.jobs.taskValues.set(key="refreshed", value=refreshed)  # noqa: F821
