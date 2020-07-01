import pandas as pd
import pytest
from hcrystalball.wrappers import (
    ExponentialSmoothingWrapper,
    SimpleSmoothingWrapper,
    HoltSmoothingWrapper,
)


@pytest.mark.parametrize("X_y_optional", [("just_X"), ("")], indirect=["X_y_optional"])
@pytest.mark.parametrize(
    "model_type", [ExponentialSmoothingWrapper, HoltSmoothingWrapper, SimpleSmoothingWrapper],
)
def test_smoothing_transform_data_to_tsmodel_input_format(X_y_optional, model_type):

    X, y = X_y_optional
    smoothing = model_type()
    endog = smoothing._transform_data_to_tsmodel_input_format(X, y)
    if y is not None:
        assert isinstance(endog, pd.Series)
        assert endog.shape[0] == y.shape[0]
    else:
        assert endog == X.shape[0]
