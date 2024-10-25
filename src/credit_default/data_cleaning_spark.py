import os
from typing import Any, Dict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from pyspark.sql import SparkSession

from credit_default.utils import TargetConfig

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
        config (Dict[str, Any]): Configuration dictionary containing preprocessing settings
        df (pd.DataFrame): DataFrame containing the data to be processed
        target_config (TargetConfig): Configuration for target variable
    """

    def __init__(self, filepath: str, config: Dict[str, Any], spark: SparkSession):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): Path to the CSV file containing the data
            config (Dict[str, Any]): Configuration dictionary containing preprocessing settings

        Raises:
            FileNotFoundError: If data file doesn't exist
            Exception: If data cleaning fails
        """
        # self._validate_file_exists(filepath)
        self.config = config
        self.spark = spark
        self._validate_and_setup_config()
        self.df = self._load_data(filepath)
        self._validate_dataframe()

    # @staticmethod
    # def _validate_file_exists(filepath: str) -> None:
    #     """
    #     Validates that the input file exists.

    #     Args:
    #         filepath (str): Path to the CSV file containing the data

    #     Raises:
    #         FileNotFoundError: If file does not exist
    #     """
    #     if not os.path.exists(filepath):
    #         logger.error(f"File not found: {filepath}")
    #         raise FileNotFoundError(f"The file {filepath} does not exist")

    def _validate_and_setup_config(self) -> None:
        """
        Validates configuration and sets up target configuration.

        Raises:
            Exception: If data cleaning fails
        """
        try:
            logger.info("Validating configuration settings")
            self._validate_config_structure()
            self._setup_target_config()
        except Exception as e:
            raise Exception(f"Configuration validation failed: {str(e)}") from e

    def _validate_config_structure(self) -> None:
        """Validates the structure of the configuration dictionary."""
        required_keys = ["columns_to_drop", "target"]
        missing_keys = [key for key in required_keys if key not in self.config]

        if missing_keys:
            raise Exception(f"Missing required keys: {', '.join(missing_keys)}")

        if not isinstance(self.config["columns_to_drop"], list):
            raise Exception("'columns_to_drop' must be a list")

        if not isinstance(self.config["target"], list) or not self.config["target"]:
            raise Exception("'target' must be a non-empty list")

        if "name" not in self.config["target"][0]:
            raise Exception("Target configuration must contain 'name' key")

    def _setup_target_config(self) -> None:
        """Sets up target configuration from config dictionary."""
        self.target_config = TargetConfig(name=self.config["target"][0]["name"])

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
        except Exception as e:
            raise Exception(f"Unexpected error loading data: {str(e)}") from e

    def _validate_dataframe(self) -> None:
        """
        Validates the loaded DataFrame structure and content.

        Raises:
            Exception: If DataFrame validation fails
        """
        logger.info("Validating DataFrame structure and content")

        # Check for required columns
        self._validate_columns()

        # Check for null values
        null_counts = self.df.isnull().sum()
        if null_counts.any():
            logger.warning(f"Found null values:\n{null_counts[null_counts > 0]}")

        # Check data types
        self._validate_data_types()

    def _validate_columns(self) -> None:
        """
        Validates that required columns exist in the DataFrame.

        Raises:
            Exception: If DataFrame validation fails
        """
        columns_to_check = self.config.get("columns_to_drop", []) + [self.target_config.name]
        missing_columns = [col for col in columns_to_check if col not in self.df.columns]

        if missing_columns:
            raise Exception(f"Missing required columns: {', '.join(missing_columns)}")

    def _validate_data_types(self) -> None:
        """Validates data types of key columns.

        Raises:
            Exception: If DataFrame validation fails
        """
        try:
            # Ensure target variable is numeric
            target_col = self.target_config.name
            if not np.issubdtype(self.df[target_col].dtype, np.number):
                raise Exception(f"Target column '{target_col}' must be numeric")

        except Exception as e:
            raise Exception(f"Data type validation failed: {str(e)}") from e

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
            self._rename_target_column()
            self._capitalize_columns()
            self._correct_unknown_values()

            # Validate final dataset
            self._validate_preprocessed_data()

            logger.info("Data preprocessing completed successfully")
            return self.df

        except Exception as e:
            raise Exception(f"Preprocessing failed: {str(e)}") from e

    def _drop_columns(self) -> None:
        """Removes specified columns from the DataFrame."""
        columns_to_drop = self.config.get("columns_to_drop", [])
        logger.info(f"Dropping columns: {', '.join(columns_to_drop)}")
        self.df = self.df.drop(columns=columns_to_drop)

    def _rename_target_column(self) -> None:
        """Renames the target column."""
        logger.info(f"Renaming target column '{self.target_config.name}' to '{self.target_config.new_name}'")
        self.df = self.df.rename(columns={self.target_config.name: self.target_config.new_name})

    def _capitalize_columns(self) -> None:
        """Capitalizes column names."""
        logger.info("Capitalizing column names")
        self.df.columns = self.df.columns.str.capitalize()

    def _correct_unknown_values(self) -> None:
        """Corrects unknown values in specified columns."""
        logger.info("Correcting unknown values for Education, Marriage and Pay columns")

        corrections = {"Education": {0: 4, 5: 4, 6: 4}, "Marriage": {0: 3}, "Pay": {-1: 0, -2: 0}}

        self._apply_corrections(corrections)

    def _apply_corrections(self, corrections: Dict[str, Dict[Any, Any]]) -> None:
        """
        Applies value corrections to specified columns.

        Args:
            corrections: Dictionary mapping column prefixes to value replacement dictionaries
        """
        for col_prefix, replacement_dict in corrections.items():
            columns = [col for col in self.df.columns if col.startswith(col_prefix)]
            for col in columns:
                self._replace_values(col, replacement_dict)

    def _replace_values(self, column: str, replacement_dict: Dict[Any, Any]) -> None:
        """
        Replaces values in a specified column based on a replacement dictionary.

        Args:
            column: Column name to perform replacements on
            replacement_dict: Dictionary mapping old values to new values
        """
        if column in self.df.columns:
            self.df[column] = self.df[column].replace(replacement_dict)
        else:
            logger.warning(f"Column '{column}' not found in the data")

    def _validate_preprocessed_data(self) -> None:
        """
        Validates the preprocessed data before returning.

        Raises:
            Exception: If validation fails
        """
        if self.df.empty:
            raise Exception("Preprocessing resulted in empty DataFrame")

        # Validate target column
        target_col = self.target_config.new_name
        if target_col not in self.df.columns:
            raise Exception(f"Target column '{target_col}' missing after preprocessing")

        # Check for unexpected null values
        if self.df.isnull().any().any():
            raise Exception("Unexpected null values found after preprocessing")


# if __name__ == "__main__":
#     # Set up logging
#     setup_logging(CLEANING_LOGS)

#     try:
#         # Load configuration
#         config = load_config(CONFIG)
#         logger.info(f"Loaded configuration from {config}")

#         # Create and run data cleaner
#         data_cleaner = DataCleaning(FILEPATH_DATABRICKS, config)
#         cleaned_data = data_cleaner.preprocess_data()

#         # Log results
#         logger.info("Data cleaning completed successfully:")
#         logger.info(f"Final data shape: {cleaned_data.shape}")
#         logger.info(f"Final columns: {cleaned_data.columns.tolist()}")
#         logger.info(f"Sample of cleaned data:\n{cleaned_data.head().to_string()}")

#     except ValueError as e:
#         logger.error(f"Data cleaning failed: {str(e)}")
#         raise
#     except Exception as e:
#         logger.error(f"Unexpected error: {str(e)}")
#         raise

#     logger.info("Data cleaning script completed successfully")
