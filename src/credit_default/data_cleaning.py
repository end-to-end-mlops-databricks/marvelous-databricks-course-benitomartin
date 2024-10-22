import os
import sys
from typing import Any, Dict

import pandas as pd
import yaml
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
CLEANING_LOGS = os.environ["CLEANING_LOGS"]
CONFIG = os.environ["CONFIG"]


class DataCleaning:
    """A class for cleaning and preprocessing credit default data."""

    def __init__(self, filepath: str, config: Dict[str, Any]):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (dict): The configuration dictionary containing preprocessing settings.
        """
        self.config = config
        self.validate_config()
        self.df = self.load_data(filepath)
        self.validate_columns()

    @staticmethod
    def load_data(filepath: str) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Args:
            filepath (str): The path to the CSV file to be loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.

        Raises:
            FileNotFoundError: If the file does not exist.
            pd.errors.EmptyDataError: If the file is empty or unreadable.
        """
        logger.info(f"Loading data from {filepath}")
        if not os.path.exists(filepath):
            logger.error(f"File {filepath} does not exist")
            raise FileNotFoundError(f"File {filepath} does not exist")
        try:
            df = pd.read_csv(filepath)
            logger.info("File loaded successfully")
            return df
        except pd.errors.EmptyDataError as e:
            logger.error(f"File is empty or unreadable: {e}")
            raise

    def validate_config(self) -> None:
        """
        Validates the configuration settings. Checks for the existence of necessary keys
        like 'columns_to_drop' and 'target' in the config dictionary.

        Raises:
            KeyError: If required keys are missing from the config.
        """
        logger.info("Validating configuration settings")
        required_keys = ["columns_to_drop", "target"]

        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            logger.error(f"Missing keys in config: {', '.join(missing_keys)}")
            raise KeyError(f"Missing keys in config: {', '.join(missing_keys)}")

        if not isinstance(self.config["target"], list) or "name" not in self.config["target"][0]:
            logger.error("Invalid target configuration. Must be a list with 'name' key.")
            raise KeyError("Invalid target configuration. Must be a list with 'name' key.")

    def validate_columns(self) -> None:
        """
        Validates that the required columns specified in the configuration exist in the DataFrame.

        Raises:
            KeyError: If any column specified in the config does not exist in the data.
        """
        logger.info("Validating if required columns exist in the DataFrame")
        columns_to_check = self.config.get("columns_to_drop", []) + [self.config["target"][0]["name"]]

        missing_columns = [col for col in columns_to_check if col not in self.df.columns]

        if missing_columns:
            logger.error(f"Missing columns in the data: {', '.join(missing_columns)}")
            raise KeyError(f"Missing columns in the data: {', '.join(missing_columns)}")

    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocesses the data by performing several cleaning steps.

        Returns:
            pd.DataFrame: The cleaned DataFrame after preprocessing.
        """
        logger.info("Starting data preprocessing")

        self._drop_columns()
        self._rename_target_column()
        self._capitalize_columns()
        self._correct_unknown_values()

        logger.info("Data preprocessing completed")
        return self.df

    def _drop_columns(self) -> None:
        """Removes specified columns from the DataFrame."""
        columns_to_drop = self.config.get("columns_to_drop", [])
        logger.info(f"Dropping columns: {', '.join(columns_to_drop)}")
        self.df = self.df.drop(columns=columns_to_drop)

    def _rename_target_column(self) -> None:
        """Renames the target column."""
        target_name = self.config["target"][0]["name"]
        new_target_name = "Default"
        logger.info(f"Renaming target column '{target_name}' to '{new_target_name}'")
        self.df = self.df.rename(columns={target_name: new_target_name})

    def _capitalize_columns(self) -> None:
        """Capitalizes column names."""
        logger.info("Capitalizing column names")
        self.df.columns = self.df.columns.str.capitalize()

    def _correct_unknown_values(self) -> None:
        """Corrects unknown values in specified columns."""
        logger.info("Correcting unknown values for Education, Marriage and Pay")

        self._replace_values("Education", {0: 4, 5: 4, 6: 4})
        self._replace_values("Marriage", {0: 3})

        pay_columns = ["Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6"]
        for col in pay_columns:
            self._replace_values(col, {-1: 0, -2: 0})

    def _replace_values(self, column: str, replacement_dict: Dict[Any, Any]) -> None:
        """
        Replaces values in a specified column based on a replacement dictionary.

        Args:
            column (str): The name of the column to perform replacements on.
            replacement_dict (Dict[Any, Any]): A dictionary mapping old values to new values.
        """
        if column in self.df.columns:
            self.df[column] = self.df[column].replace(replacement_dict)
        else:
            logger.warning(f"'{column}' column not found in the data")


if __name__ == "__main__":
    # # Configure logger
    # Remove the default logger
    logger.remove()

    # Add logger to file with rotation
    logger.add(CLEANING_LOGS, level="DEBUG", rotation="500 MB")

    # Add logger to console (stdout) with DEBUG level
    logger.add(sys.stdout, level="DEBUG")

    # Load configuration from YAML file
    with open(CONFIG, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Test configuration: {config}")

    # Data path
    filepath = FILEPATH
    logger.info(f"Data filepath: {filepath}")

    try:
        # Create an instance of DataCleaning
        data_cleaner = DataCleaning(filepath, config)
        logger.info("DataCleaning instance created successfully")

        # Preprocess the data
        cleaned_data = data_cleaner.preprocess_data()
        logger.info("Data preprocessing completed")

        # Log the first few rows of the cleaned data
        logger.info("First few rows of cleaned data\n" + cleaned_data.head().to_string())

        # Log basic information about the cleaned data
        logger.info(f"Cleaned data shape: {cleaned_data.shape}")
        logger.info(f"Cleaned data columns: {cleaned_data.columns.tolist()}")

    except Exception as e:
        logger.error(f"An error occurred during the test: {str(e)}")

    logger.info("DataCleaning test completed")
