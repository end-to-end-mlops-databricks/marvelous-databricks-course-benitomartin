import os
import sys

import yaml
from dotenv import load_dotenv
from loguru import logger

from src.credit_default.data_preprocessing import DataPreprocessor

# Load environment variables
load_dotenv()

FILEPATH = os.environ["FILEPATH"]
CONFIG = os.environ["CONFIG"]
PIPELINE_LOGS = os.environ["PIPELINE_LOGS"]


class CreditDefaultPipeline:
    """
    A class to manage the pipeline for credit default data preprocessing.

    This class handles loading configuration settings, initializing the data
    preprocessing steps, and running the pipeline to process the data for
    further analysis or modeling.
    """

    def __init__(self, config_path: str, data_path: str):
        """
        Initialize the CreditDefaultPipeline.

        Args:
            config_path (str): The file path to the configuration YAML file.
            data_path (str): The file path to the CSV data file to be processed.
        """
        self.config_path = config_path
        self.data_path = data_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """
        Load the configuration settings from the specified YAML file.

        Returns:
            dict: The configuration settings loaded from the YAML file.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
            yaml.YAMLError: If there is an error parsing the YAML file.
        """
        logger.info("Loading configuration from {}", self.config_path)
        if not os.path.exists(self.config_path):
            logger.error("Configuration file not found: {}", self.config_path)
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)
            logger.info("Configuration loaded:\n{}", yaml.dump(config, default_flow_style=False))
            return config
        except yaml.YAMLError as e:
            logger.error("Error parsing YAML file: {}", str(e))
            raise

    def run(self):
        """
        Run the data preprocessing pipeline.

        This method initializes the DataPreprocessor, processes the data,
        and logs the shapes of the processed features and target.
        """
        logger.info("Initializing DataPreprocessor with data path: {}", self.data_path)

        try:
            # Initialize DataPreprocessor
            data_preprocessor = DataPreprocessor(self.data_path, self.config)

            # Get processed data
            X, y, preprocessor = data_preprocessor.get_processed_data()

            # Log shapes of processed data
            logger.info("Features shape: {}", X.shape)
            logger.info("Target shape: {}", y.shape)
            logger.info("Preprocessor: {}", preprocessor)
        except Exception as e:
            logger.error("An error occurred during the data preprocessing pipeline: {}", str(e))
            sys.exit(1)


if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(PIPELINE_LOGS, level="DEBUG", rotation="500 MB")
    logger.add(sys.stdout, level="DEBUG")

    try:
        pipeline = CreditDefaultPipeline(CONFIG, FILEPATH)
        pipeline.run()
    except Exception as e:
        logger.error("An error occurred in the main execution: {}", str(e))
        sys.exit(1)

    logger.info("Credit default pipeline completed successfully")
