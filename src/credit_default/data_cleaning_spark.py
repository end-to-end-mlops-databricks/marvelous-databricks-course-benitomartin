## Databricks notebook source
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from credit_default.utils import Config, Target

# Load environment variables
load_dotenv()

spark = SparkSession.builder.getOrCreate()

FILEPATH_DATABRICKS = os.environ["FILEPATH_DATABRICKS"]
CLEANING_LOGS = os.environ["CLEANING_LOGS"]
CONFIG_DATABRICKS = os.environ["CONFIG_DATABRICKS"]


class DataCleaning:
    """
    A class for cleaning and preprocessing credit default data.

    Attributes:
        config (Config): Configuration model containing preprocessing settings
        df (pd.DataFrame): DataFrame containing the data to be processed
        target_config (Target): Configuration for target variable
    """

    def __init__(self, filepath: str, config: Config, spark: SparkSession):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): Path to the CSV file containing the data
            config (Config): Configuration model containing preprocessing settings

        Raises:
            Exception: If data cleaning fails
        """
        self.config = config
        self.spark = spark
        self.df = self._load_data(filepath)
        self._setup_target_config()
        self._validate_dataframe()

    def _setup_target_config(self) -> None:
        """Sets up target configuration from config."""
        target_info = self.config.target[0]
        self.target_config = Target(name=target_info.name, dtype=target_info.dtype, new_name=target_info.new_name)

    @staticmethod
    def _load_data(filepath: str) -> pd.DataFrame:
        """
        Loads and validates the input data.

        Args:
            filepath (str): Path to the CSV file

        Returns:
            pd.DataFrame: Loaded DataFrame

        Raises:
            Exception: If data loading or validation fails
        """
        try:
            logger.info(f"Loading data from {filepath}")

            df = spark.read.csv(FILEPATH_DATABRICKS, header=True, inferSchema=True).toPandas()

            if df.empty:
                raise Exception("Loaded DataFrame is empty")
            return df
        except pd.errors.EmptyDataError as e:
            raise Exception(f"Failed to load data: {str(e)}") from e

    def _validate_dataframe(self) -> None:
        """
        Validates the loaded DataFrame structure and content.

        Raises:
            Exception: If DataFrame validation fails
        """
        columns_to_check = self.config.columns_to_drop + [self.target_config.name]
        missing_columns = [col for col in columns_to_check if col not in self.df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    def _validate_columns(self) -> None:
        """
        Validates that required columns exist in the DataFrame.

        Raises:
            Exception: If DataFrame validation fails
        """
        columns_to_check = self.config.columns_to_drop + [self.target_config.name]
        missing_columns = [col for col in columns_to_check if col not in self.df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    def _validate_data_types(self) -> None:
        """Validates data types of key columns.

        Raises:
            Exception: If DataFrame validation fails
        """
        target_col = self.target_config.name
        if not np.issubdtype(self.df[target_col].dtype, np.number):
            raise Exception(f"Target column '{target_col}' must be numeric")

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses the data by performing several cleaning steps.

        Returns:
            pd.DataFrame: The cleaned DataFrame after preprocessing.

        Raises:
            Exception: If preprocessing fails
        """
        try:
            logger.info("Starting data preprocessing")
            self._drop_columns()
            self._rename_and_capitalize_columns()
            self._apply_value_corrections()
            self._convert_int_to_float()
            self._validate_preprocessed_data()
            logger.info("Data preprocessing completed successfully")
            return self.df
        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}") from e

    def _drop_columns(self) -> None:
        """Removes specified columns from the DataFrame."""
        columns_to_drop = getattr(self.config, "columns_to_drop", [])
        if columns_to_drop:
            self.df.drop(
                columns=[col for col in columns_to_drop if col in self.df.columns], inplace=True, errors="ignore"
            )
            logger.info(f"Dropped columns: {', '.join(columns_to_drop)}")

    def _rename_and_capitalize_columns(self) -> None:
        """Renames and capitalizes key columns."""
        self.df.rename(columns={self.target_config.name: self.target_config.new_name}, inplace=True)
        self.df.columns = [col.capitalize() if col else col for col in self.df.columns]
        logger.info("Renamed and capitalized columns")

    def _convert_int_to_float(self) -> None:
        """Converts integer columns to float to avoid schema enforcement errors with nulls."""
        logger.info("Converting integer columns to float (due to spark warning)")
        for col in self.df.select_dtypes(include="integer").columns:
            self.df[col] = self.df[col].astype(float)

    def _apply_value_corrections(self) -> None:
        """Corrects unknown values in specified columns."""
        logger.info("Applying value corrections for Education, Marriage, and Pay columns")
        corrections = getattr(
            self.config,
            "value_corrections",
            {
                "Education": {0: 4, 5: 4, 6: 4},
                "Marriage": {0: 3},
                "Pay": {-1: 0, -2: 0},
            },
        )
        for col_prefix, replacement_dict in corrections.items():
            columns = [col for col in self.df.columns if col.startswith(col_prefix)]
            for col in columns:
                self.df[col] = self.df[col].replace(replacement_dict)

    def _validate_preprocessed_data(self) -> None:
        """Validates the preprocessed data before returning."""
        if self.df.empty:
            raise Exception("Preprocessing resulted in an empty DataFrame")
        target_col = self.target_config.new_name
        if target_col not in self.df.columns:
            raise Exception(f"Target column '{target_col}' missing after preprocessing")
        if self.df.isnull().any().any():
            raise Exception("Unexpected null values found after preprocessing")


# if __name__ == "__main__":
#     # Set up logging
#     setup_logging(CLEANING_LOGS)

#     try:
#         # Load configuration
#         config = load_config(CONFIG_DATABRICKS)  # Returns Config instance
#         logger.info(f"Loaded configuration from {CONFIG_DATABRICKS}")

#         # Create and run data cleaner
#         data_cleaner = DataCleaning(FILEPATH_DATABRICKS, config, spark)
#         cleaned_data = data_cleaner.preprocess_data()

#         # Log results
#         logger.info("Data cleaning completed successfully")
#         logger.info(f"Final data shape: {cleaned_data.shape}")
#         logger.info(f"Final columns: {cleaned_data.columns.tolist()}")
#         logger.info(f"Sample of cleaned data:\n{cleaned_data.head().to_string()}")

#     except ValidationError as e:
#         logger.error(f"Configuration validation error: {e}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise

#     logger.info("Data cleaning script completed successfully")
