import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

from src.credit_default.data_preprocessing import DataPreprocessor


@pytest.fixture
def real_data():
    """Load real data for testing."""
    return pd.read_csv("data/data.csv")


@pytest.fixture
def config():
    """Load configuration from the YAML file for testing."""
    with open("project_config.yml", "r") as f:
        config_data = yaml.safe_load(f)
    return config_data


def test_data_preprocessor(real_data, config):
    """Test the DataPreprocessor class for preprocessing and scaling."""
    data_preprocessor = DataPreprocessor("data/data.csv", config)

    # Check that the cleaned data is as expected
    assert data_preprocessor.cleaned_data.equals(data_preprocessor.data_cleaning.df)

    # Check that features and target are correctly separated
    assert data_preprocessor.X.shape[0] == data_preprocessor.cleaned_data.shape[0]
    assert data_preprocessor.y.shape[0] == data_preprocessor.cleaned_data.shape[0]

    # Check that the correct features are used for scaling
    assert set(data_preprocessor.features_robust) == {
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
    }

    # Check that the preprocessor is of type ColumnTransformer
    assert isinstance(data_preprocessor.preprocessor, ColumnTransformer)

    # Check that the robust scaler is applied to the correct features
    transformers = data_preprocessor.preprocessor.transformers
    assert transformers[0][0] == "robust_scaler"
    assert isinstance(transformers[0][1], RobustScaler)
    assert set(transformers[0][2]) == set(data_preprocessor.features_robust)

    # Check if the output shape after scaling is as expected
    X_scaled, y, preprocessor = data_preprocessor.get_processed_data()
    assert X_scaled.shape[0] == data_preprocessor.X.shape[0]
    assert y.shape[0] == data_preprocessor.y.shape[0]
