import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from fbprophet import Prophet
from hcrystalball.wrappers import ProphetWrapper


@pytest.mark.parametrize(
    "X_with_holidays, extra_holidays",
    [
        ("", {"New year": {"prior_scale": 100}, "Whit Monday": {"lower_window": 2, "prior_scale": 10}}),
        ("", {"New year": {"prior_scale": 100, "lower_window": 1}}),
        ("", None),
    ],
    indirect=["X_with_holidays"],
)
def test_prophet_adjust_holidays(X_with_holidays, extra_holidays):

    prophet = ProphetWrapper(extra_holidays=extra_holidays)
    prophet.model = prophet._init_tsmodel(Prophet)
    X = prophet._adjust_holidays(X_with_holidays)
    holidays = prophet.model.holidays
    assert_frame_equal(X, X_with_holidays.drop(columns="holiday"))
    assert isinstance(holidays, pd.DataFrame)
    assert "ds" in holidays.columns.tolist()
    assert "holiday" in holidays.columns.tolist()
    max_holiday_len = (
        0
        if prophet.extra_holidays is None
        else max([len(prophet.extra_holidays[key]) for key in prophet.extra_holidays.keys()], default=0,)
    )
    assert holidays.shape[1] == len(["ds", "holiday"]) + max_holiday_len
    if extra_holidays:
        for holiday_name, holiday_params in extra_holidays.items():
            for params_key, params_values in holiday_params.items():
                assert (
                    holidays.loc[holidays["holiday"] == holiday_name, params_key].values[0] == params_values
                )


@pytest.mark.parametrize(
    "X_y_optional, additional_col",
    [("just_X", None), ("just_X", True), ("", None), ("", True)],
    indirect=["X_y_optional"],
)
def test_prophet_transform_data_to_tsmodel_input_format(X_y_optional, additional_col):

    X, y = X_y_optional
    if additional_col:
        X["additiona_col"] = 1
    prophet = ProphetWrapper()
    result = prophet._transform_data_to_tsmodel_input_format(X, y)
    assert "ds" in result.columns.tolist()
    assert result["ds"].dtype.kind == "M"
    assert result.shape[0] == X.shape[0]
    col_count = 1
    if y is not None:
        assert "y" in result.columns.tolist()
        col_count += 1
    if additional_col:
        assert "additiona_col" in result.columns.tolist()
        col_count += 1
    assert result.shape[1] == col_count
