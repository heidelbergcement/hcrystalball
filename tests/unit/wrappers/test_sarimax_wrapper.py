import pytest
import pandas as pd
import numpy as np
from hcrystalball.wrappers import SarimaxWrapper


@pytest.mark.parametrize("X_with_holidays", [(""), (""), ("")], indirect=["X_with_holidays"])
def test_sarimax_adjust_holidays(X_with_holidays):

    sarimax = SarimaxWrapper(order=(1, 1, 0))
    result = sarimax._adjust_holidays(X_with_holidays)
    assert isinstance(result, pd.DataFrame)
    assert "_holiday_DE" in result.columns.tolist()
    assert "_holiday_DE" in result.select_dtypes(include=[np.bool]).columns
    assert (result["_holiday_DE"] == "").sum() == 0
    assert X_with_holidays[X_with_holidays["_holiday_DE"] != ""].shape[0] == result["_holiday_DE"].sum()


@pytest.mark.parametrize(
    "X_y_optional, additional_col",
    [("just_X", None), ("just_X", True), ("", None), ("", True)],
    indirect=["X_y_optional"],
)
def test_sarimax_transform_data_to_tsmodel_input_format(X_y_optional, additional_col):

    X, y = X_y_optional
    if additional_col:
        X["additional_col"] = 1
    sarimax = SarimaxWrapper(order=(1, 1, 0))
    endog, exog = sarimax._transform_data_to_tsmodel_input_format(X, y)
    if y is not None:
        assert isinstance(endog, pd.Series)
        assert endog.shape[0] == y.shape[0]
    if additional_col:
        assert isinstance(exog, np.ndarray)
        assert exog.shape[1] == 1
        assert exog.shape[0] == X.shape[0]
    if additional_col is None:
        assert exog is None


@pytest.mark.parametrize("X_y_linear_trend", [("more_cols_freq_D")], indirect=["X_y_linear_trend"])
def test_autoarima_init(X_y_linear_trend):

    X, y = X_y_linear_trend
    sarimax = SarimaxWrapper(init_with_autoarima=True, autoarima_dict={"D": 1, "m": 2})
    sarimax.fit(X[:-10], y[:-10])
    first_params = sarimax.get_params()
    sarimax.fit(X[:-9], y[:-9])
    second_params = sarimax.get_params()
    assert first_params == second_params


@pytest.mark.parametrize(
    "X_y_linear_trend, init_params",
    [
        ("more_cols_freq_D", {"order": (1, 1, 1), "seasonal_order": (0, 0, 0, 2)}),
        (
            "more_cols_freq_D",
            {"init_with_autoarima": True, "autoarima_dict": {"D": 1, "m": 2}},
        ),
        (
            "more_cols_freq_D",
            {"always_search_model": True, "autoarima_dict": {"D": 1, "m": 2}, "init_with_autoarima": True},
        ),
    ],
    indirect=["X_y_linear_trend"],
)
def test_deserialization(X_y_linear_trend, init_params):

    X, y = X_y_linear_trend
    sarimax = SarimaxWrapper(**init_params)
    sarimax.fit(X[:-10], y[:-10])
    first_params = sarimax.get_params()

    re_sarimax = SarimaxWrapper(**first_params)
    re_sarimax.fit(X[:-10], y[:-10])
    re_first_params = re_sarimax.get_params()

    assert first_params == re_first_params
