import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from hcrystalball.exceptions import InsufficientDataLengthError
from hcrystalball.wrappers import get_sklearn_wrapper


@pytest.mark.parametrize(
    "X_y_linear_trend, horizon, exp_error",
    [
        ("more_cols_freq_D", 10, None),
        ("more_cols_freq_D", 48, None),
        ("more_cols_freq_D", 49, InsufficientDataLengthError),
    ],
    indirect=["X_y_linear_trend"],
)
def test_sklearn_wrapper_overal(X_y_linear_trend, horizon, exp_error):
    CONSTANT = 50
    X, y = X_y_linear_trend
    model = get_sklearn_wrapper(DummyRegressor, lags=3, strategy="constant", constant=CONSTANT)
    model.fit(X[:-horizon], y[:-horizon])

    if exp_error is not None:
        with pytest.raises(exp_error):
            preds = model.predict(X[-horizon:])
    else:
        preds = model.predict(X[-horizon:])
        assert all(preds == CONSTANT)
