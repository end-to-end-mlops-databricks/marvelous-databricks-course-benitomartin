import sys
from typing import Any, Dict, List, Literal

import yaml
from loguru import logger
from pydantic import BaseModel, Field, ValidationError


class NumFeature(BaseModel):
    name: str
    dtype: Literal["float64", "int64"]


class Target(BaseModel):
    name: str
    dtype: Literal["float64", "int64"]
    new_name: str


class Features(BaseModel):
    robust: List[str]


class Config(BaseModel):
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any] = Field(description="Parameters for model training")
    num_features: List[NumFeature]
    target: List[Target]
    columns_to_drop: List[str]
    features: Features


def setup_logging(log_file: str, log_level: str = "DEBUG") -> None:
    """
    Sets up logging configuration with rotation.

    Args:
        log_file (str): Path to the log file
        log_level (str, optional): Logging level to use. Defaults to "DEBUG".
    """
    # Remove default logger
    logger.remove()

    # Add file logger with rotation
    logger.add(log_file, level=log_level, rotation="500 MB")

    # Add stdout logger if requested
    logger.add(
        sys.stdout,
        level=log_level,
    )


def load_config(config_path: str) -> Config:
    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
        config = Config(**config_data)  # Pydantic validation
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file: {str(e)}")
        raise
    except ValidationError as e:
        logger.error(f"Validation error in configuration: {e}")
        raise


# import sys
# from dataclasses import dataclass
# from typing import Any, Dict

# import yaml  # Import yaml
# from loguru import logger


# @dataclass
# class TargetConfig:
#     """
#     Configuration for target variable.

#     Attributes:
#         name (str): Original name of the target column
#         new_name (str): New name to be used for the target column
#     """

#     name: str
#     new_name: str = "Default"


# def setup_logging(log_file: str, log_level: str = "DEBUG") -> None:
#     """
#     Sets up logging configuration with rotation.

#     Args:
#         log_file (str): Path to the log file
#         log_level (str, optional): Logging level to use. Defaults to "DEBUG".
#     """
#     # Remove default logger
#     logger.remove()

#     # Add file logger with rotation
#     logger.add(log_file, level=log_level, rotation="500 MB")

#     # Add stdout logger if requested
#     logger.add(
#         sys.stdout,
#         level=log_level,
#     )


# def load_config(config_path: str) -> Dict[str, Any]:
#     """
#     Loads configuration from a YAML file.

#     Args:
#         config_path (str): Path to the configuration file.

#     Returns:
#         Dict[str, Any]: Loaded configuration dictionary.

#     Raises:
#         FileNotFoundError: If the configuration file doesn't exist.
#         yaml.YAMLError: If there is an error parsing the YAML file.
#     """
#     try:
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)
#         logger.info(f"Loaded configuration from {config_path}")
#         return config
#     except FileNotFoundError:
#         logger.error(f"Configuration file not found: {config_path}")
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f"Error parsing YAML configuration file: {str(e)}")
#         raise
