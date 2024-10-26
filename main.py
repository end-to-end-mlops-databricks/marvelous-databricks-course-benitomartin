import os
import sys
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger

from credit_default.data_preprocessing import DataPreprocessor
from credit_default.model_training import ModelTrainer
from credit_default.utils import load_config, setup_logging

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
        self.config_path: str = config_path
        self.data_path: str = data_path
        self.config: Dict = load_config(self.config_path)
        self.learning_rate = self.config.parameters["learning_rate"]
        self.random_state = self.config.parameters["random_state"]

    def run(self) -> None:
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

            # Train and evaluate the model
            self.train_and_evaluate(X, y, preprocessor)

        except Exception as e:
            logger.error("An error occurred during the data preprocessing pipeline: {}", str(e))
            sys.exit(1)

    def train_and_evaluate(self, X: Any, y: Any, preprocessor: Any) -> None:
        """
        Train the model using the preprocessed data and evaluate it.

        Args:
            X (Any): The features DataFrame.
            y (Any): The target Series.
            preprocessor (Any): The preprocessor for scaling.
        """
        logger.info("Starting model training and evaluation process")

        try:
            # Initialize ModelTrainer
            model_trainer = ModelTrainer(
                X, y, preprocessor, learning_rate=self.learning_rate, random_state=self.random_state
            )

            model_trainer.train()

            # Evaluate the model
            (auc_val, conf_matrix_val, class_report_val), (auc_test, conf_matrix_test, class_report_test) = (
                model_trainer.evaluate()
            )

            # Log evaluation results
            logger.info("Validation AUC: {}", auc_val)
            logger.info("\nValidation Confusion Matrix:\n {}", conf_matrix_val)
            logger.info("\nValidation Classification Report:\n {}", class_report_val)

            logger.info("\nTest AUC: {}", auc_test)
            logger.info("\nTest Confusion Matrix:\n {}", conf_matrix_test)
            logger.info("\nTest Classification Report:\n {}", class_report_test)

            logger.info("Model training and evaluation process completed successfully")
        except Exception as e:
            logger.error("An error occurred during the model training and evaluation: {}", str(e))
            sys.exit(1)


if __name__ == "__main__":
    # Configure logger using setup_logging
    setup_logging(PIPELINE_LOGS)

    try:
        pipeline = CreditDefaultPipeline(CONFIG, FILEPATH)
        pipeline.run()
    except Exception as e:
        logger.error("An error occurred in the main execution: {}", str(e))
        sys.exit(1)

    logger.info("Credit default pipeline completed successfully")
