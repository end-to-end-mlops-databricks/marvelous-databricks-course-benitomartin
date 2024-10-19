import os

import pandas as pd
from loguru import logger


class DataCleaning:
    """A class for cleaning and preprocessing credit default data.

    Attributes:
        df (pd.DataFrame): The DataFrame containing the loaded data.
        config (dict): The configuration dictionary containing preprocessing settings.
        X (pd.DataFrame or None): Features DataFrame after preprocessing, initialized to None.
        y (pd.Series or None): Target Series after preprocessing, initialized to None.
        preprocessor (sklearn.compose.ColumnTransformer or None): ColumnTransformer for preprocessing, initialized to None.
    """

    def __init__(self, filepath, config):
        """
        Initializes the DataCleaning class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (dict): The configuration dictionary containing preprocessing settings.
        """
        self.config = config
        self.df = self.load_data(filepath)
        self.X = None
        self.y = None
        self.preprocessor = None
        self.validate_config()

    def load_data(self, filepath):
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
        logger.info("Loading data from {}", filepath)
        if not os.path.exists(filepath):
            logger.error("File {} does not exist", filepath)
            raise FileNotFoundError(f"File {filepath} does not exist")
        try:
            df = pd.read_csv(filepath)
            logger.info("File loaded successfully")
            return df
        except pd.errors.EmptyDataError as e:
            logger.error("File is empty or unreadable: {}", e)
            raise pd.errors.EmptyDataError("File is empty or unreadable") from e

    def validate_config(self):
        """
        Validates the configuration settings. Checks for the existence of necessary keys
        like 'columns_to_drop' and 'target' in the config dictionary.

        Raises:
            KeyError: If required keys are missing from the config.
        """
        logger.info("Validating configuration settings")
        required_keys = ["columns_to_drop", "target"]

        for key in required_keys:
            if key not in self.config:
                logger.error("Missing '{}' in config", key)
                raise KeyError(f"Missing '{key}' in config")

        if not isinstance(self.config["target"], list) or "name" not in self.config["target"][0]:
            logger.error("Invalid target configuration. Must be a list with 'name' key.")
            raise KeyError("Invalid target configuration. Must be a list with 'name' key.")

    def validate_columns(self):
        """
        Validates that the required columns specified in the configuration exist in the DataFrame.

        Raises:
            KeyError: If any column specified in the config does not exist in the data.
        """
        logger.info("Validating if required columns exist in the DataFrame")
        columns_to_drop = self.config.get("columns_to_drop", [])
        target_column = self.config["target"][0]["name"]

        missing_columns = [col for col in columns_to_drop + [target_column] if col not in self.df.columns]

        if missing_columns:
            logger.error("Missing columns in the data: {}", missing_columns)
            raise KeyError(f"Missing columns in the data: {missing_columns}")

    def preprocess_data(self):
        """
        Preprocesses the data by performing several cleaning steps, including:
        - Removing specified columns.
        - Renaming the target column.
        - Capitalizing column names.
        - Correcting unknown values in specified columns.

        Returns:
            pd.DataFrame: The cleaned DataFrame after preprocessing.
        """
        logger.info("Starting data preprocessing")

        # Validate columns before processing
        self.validate_columns()

        # Remove specified columns from the config
        columns_to_drop = self.config.get("columns_to_drop", [])
        logger.info("Dropping columns: {}", columns_to_drop)
        self.df = self.df.drop(columns=columns_to_drop)

        # Rename the target column
        target_name = self.config["target"][0]["name"]
        new_target_name = "Default"
        logger.info("Renaming target column '{}' to '{}'", target_name, new_target_name)
        self.df = self.df.rename(columns={target_name: new_target_name})

        # Capitalize Columns
        logger.info("Capitalizing column names")
        self.df.columns = self.df.columns.str.capitalize()

        # Correct Unknown Values Education/Marriage/Pay
        logger.info("Correcting unknown values for Education, Marriage and Pay")

        if "Education" in self.df.columns:
            self.df["Education"] = self.df["Education"].replace({0: 4, 5: 4, 6: 4})
        else:
            logger.warning("'Education' column not found in the data")

        if "Marriage" in self.df.columns:
            self.df["Marriage"] = self.df["Marriage"].replace({0: 3})
        else:
            logger.warning("'Marriage' column not found in the data")

        columns_to_replace = ["Pay_0", "Pay_2", "Pay_3", "Pay_4", "Pay_5", "Pay_6"]
        for col in columns_to_replace:
            if col in self.df.columns:
                self.df[col] = self.df[col].replace({-1: 0, -2: 0})
            else:
                logger.warning("'{}' column not found in the data", col)

        logger.info("Data preprocessing completed")
        return self.df
