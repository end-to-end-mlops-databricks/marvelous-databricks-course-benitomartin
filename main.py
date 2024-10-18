import yaml
from loguru import logger

from src.credit_default.data_preprocessing import DataPreprocessor


class CreditDefaultPipeline:
    """A class to manage the pipeline for credit default data preprocessing.

    This class handles loading configuration settings, initializing the data
    preprocessing steps, and running the pipeline to process the data for
    further analysis or modeling.
    """

    def __init__(self, config_path, data_path):
        """Initialize the CreditDefaultPipeline.

        Args:
            config_path (str): The file path to the configuration YAML file.
            data_path (str): The file path to the CSV data file to be processed.
        """
        self.config_path = config_path
        self.data_path = data_path
        self.config = self.load_config()

    def load_config(self):
        """Load the configuration settings from the specified YAML file.

        Returns:
            dict: The configuration settings loaded from the YAML file.
        """
        logger.info("Loading configuration from {}", self.config_path)
        with open(self.config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info("Configuration loaded:\n{}", yaml.dump(config, default_flow_style=False))
        return config

    def run(self):
        """Run the data preprocessing pipeline.

        This method initializes the DataPreprocessor, processes the data,
        and logs the shapes of the processed features and target.
        """
        # Initialize DataPreprocessor
        logger.info("Initializing DataPreprocessor with {}", self.data_path)
        data_preprocessor = DataPreprocessor(self.data_path, self.config)

        # Get processed data
        X, y, preprocessor = data_preprocessor.get_processed_data()

        # Log shapes of processed data
        logger.info("Features shape: {}", X.shape)
        logger.info("Target shape: {}", y.shape)
        logger.info("Preprocessor: {}", preprocessor)


if __name__ == "__main__":
    pipeline = CreditDefaultPipeline("project_config.yml", "data/data.csv")
    pipeline.run()
