# Databricks notebook source
import os

from databricks.connect import DatabricksSession  # Import DatabricksSession
from dotenv import load_dotenv
from loguru import logger

from credit_default.data_preprocessing_spark import DataPreprocessor
from credit_default.utils import load_config, setup_logging

# Create a Databricks session
spark = DatabricksSession.builder.getOrCreate()  # Change to use DatabricksSession

# Load environment variables
load_dotenv()

FILEPATH_DATABRICKS = os.environ["FILEPATH_DATABRICKS"]
PREPROCESSING_LOGS = os.environ["PREPROCESSING_LOGS"]
# CONFIG = os.environ["CONFIG"]
CONFIG = "../../project_config.yml"
print(CONFIG)
print(FILEPATH_DATABRICKS)

# # COMMAND ----------
# df = spark.read.csv("dbfs:/Volumes/maven/default/data/data.csv", header=True, inferSchema=True).toPandas()
# df

# # COMMAND ----------
# df = spark.read.csv(FILEPATH_DATABRICKS, header=True, inferSchema=True).toPandas()
# df

# COMMAND ----------

# Load configuration from YAML file
config = load_config(CONFIG)

# Test the DataPreprocessor class
try:
    # Configure logger using setup_logging
    setup_logging(PREPROCESSING_LOGS)  # Set up logging with the log file path

    logger.info(f"Initializing DataPreprocessor with config: {config}")
    preprocessor = DataPreprocessor(FILEPATH_DATABRICKS, config, spark)
    X, y, preprocessor_model = preprocessor.get_processed_data()

    # Split data into training and test sets
    train_set, test_set = preprocessor.split_data()
    logger.info(f"Data split completed. Train shape: {train_set.shape}, Test shape: {test_set.shape}")

    # Save train and test sets to the Databricks catalog
    preprocessor.save_to_catalog(train_set, test_set, spark)
    logger.info("Train and test sets saved to catalog successfully.")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}")

logger.info("DataPreprocessor script completed")

# COMMAND ----------
