import os
from typing import Tuple

from dotenv import load_dotenv
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from credit_default.data_cleaning import DataCleaning
from credit_default.utils import load_config, setup_logging

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
PREPROCESSING_LOGS = os.environ["PREPROCESSING_LOGS"]
CONFIG = os.environ["CONFIG"]


class DataPreprocessor:
    """
    A class for preprocessing credit default data, including scaling features.

    Attributes:
        data_cleaning (DataCleaning): An instance of the DataCleaning class used for data preprocessing.
        cleaned_data (pd.DataFrame): The cleaned DataFrame after preprocessing.
        features_robust (list): List of feature names for robust scaling.
        X (pd.DataFrame): Features DataFrame after cleaning.
        y (pd.Series): Target Series after cleaning.
        preprocessor (ColumnTransformer): ColumnTransformer for scaling the features.
    """

    def __init__(self, filepath: str, config: dict):
        """
        Initializes the DataPreprocessor class.

        Args:
            filepath (str): The path to the CSV file containing the data.
            config (dict): The configuration dictionary containing preprocessing settings.
        """
        try:
            # Initialize DataCleaning to preprocess data
            logger.info("Initializing data cleaning process")
            self.data_cleaning = DataCleaning(filepath, config)
            self.cleaned_data = self.data_cleaning.preprocess_data()
            logger.info("Data cleaning process completed")

            # Define robust features for scaling
            self.features_robust = [
                "Limit_bal",
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
            ]

            # Define features and target
            self.X = self.cleaned_data.drop(columns=["Default"])
            self.y = self.cleaned_data["Default"]

            # Set up the ColumnTransformer for scaling
            logger.info("Setting up ColumnTransformer for scaling")
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("robust_scaler", RobustScaler(), self.features_robust)  # Apply RobustScaler to selected features
                ],
                remainder="passthrough",  # Keep other columns unchanged
            )
        except KeyError as e:
            logger.error(f"KeyError encountered during initialization: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"An error occurred during initialization: {str(e)}")
            raise

    def get_processed_data(self) -> Tuple:
        """
        Retrieves the processed features, target, and preprocessor.

        Returns:
            Tuple: A tuple containing:
                - pd.DataFrame: The features DataFrame.
                - pd.Series: The target Series.
                - ColumnTransformer: The preprocessor for scaling.
        """
        logger.info("Retrieving processed data and preprocessor")
        return self.X, self.y, self.preprocessor


if __name__ == "__main__":
    # Configure logger using setup_logging
    setup_logging(PREPROCESSING_LOGS)  # Set up logging with the log file path

    # Load configuration from YAML file
    config = load_config(CONFIG)

    # Test the DataPreprocessor class
    try:
        logger.info(f"Initializing DataPreprocessor with config: {config}")
        preprocessor = DataPreprocessor(FILEPATH, config)
        X, y, preprocessor_model = preprocessor.get_processed_data()

        # Log shapes of processed data
        logger.info(f"Data preprocessing completed. Shape of X: {X.shape}, Shape of y: {y.shape}")
    except Exception as e:
        logger.error(f"An error occurred during data preprocessing: {str(e)}")

    logger.info("DataPreprocessor script completed")
