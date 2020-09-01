import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal, assert_series_equal
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from hcrystalball.feature_extraction import HolidayTransformer
from hcrystalball.wrappers import ProphetWrapper
from hcrystalball.wrappers import SarimaxWrapper
from hcrystalball.wrappers import TBATSWrapper
from hcrystalball.wrappers import get_sklearn_wrapper
from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.ensemble import StackingEnsemble, SimpleEnsemble


@pytest.fixture(scope="module")
def transformers(request):
    if request.param is None:
        return None
    else:
        options = {"holiday": ("holiday", HolidayTransformer(country_code="DE"))}
        transformers = request.param.split(",")

        return [options[t] for t in transformers if t in options.keys()]


@pytest.fixture(scope="module")
def estimators():
    options = {
        "prophet": [
            (
                "prophet",
                ProphetWrapper(
                    daily_seasonality=False,
                    weekly_seasonality=False,
                    yearly_seasonality=False,
                ),
            )
        ],
        "sarimax": [("sarimax", SarimaxWrapper(order=(1, 1, 1), seasonal_order=(1, 1, 1, 2)))],
        "smoothing": [("smoothing", ExponentialSmoothingWrapper())],
        "sklearn": [("sklearn", get_sklearn_wrapper(LinearRegression))],
        "tbats": [("tbats", TBATSWrapper(use_arma_errors=False, use_box_cox=False))],
        "stacking_ensemble": [
            (
                "stacking_ensemble",
                StackingEnsemble(
                    base_learners=[
                        ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                        ExponentialSmoothingWrapper(name="smoot_exp2"),
                    ],
                    meta_model=LinearRegression(),
                ),
            )
        ],
        "simple_ensemble": [
            (
                "simple_ensemble",
                SimpleEnsemble(
                    base_learners=[
                        ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                        ExponentialSmoothingWrapper(name="smoot_exp2"),
                    ]
                ),
            )
        ],
    }

    models = []
    [models.extend(options[key]) for key in options]
    return models


@pytest.mark.parametrize(
    "X_y_linear_trend, transformers",
    [
        ("series_with_freq_D", "holiday"),
        ("ndarray_with_freq_D", "holiday"),
        # no transformers, all estimators, good data
        ("ndarray_with_freq_D", None),
        ("series_with_freq_D", None),
    ],
    indirect=["X_y_linear_trend", "transformers"],
)
def test_data_unchanged(X_y_linear_trend, transformers, estimators):
    X, y = X_y_linear_trend
    X_orig = X
    y_orig = y

    for estimator in estimators:
        fit_transform = False
        if transformers is not None and estimator != "no_estimator":
            pipe = Pipeline([("transformers", Pipeline(transformers)), estimator])
        elif transformers is None and estimator != "no_estimator":
            pipe = Pipeline([estimator])

        if fit_transform:
            pipe.fit_transform(X, y)
        else:
            pipe.fit(X, y)
            pipe.predict(X[-10:])
        assert_frame_equal(X_orig, X)

        if isinstance(y, pd.Series):
            assert_series_equal(y_orig, y)
        else:
            assert_array_equal(y_orig, y)
