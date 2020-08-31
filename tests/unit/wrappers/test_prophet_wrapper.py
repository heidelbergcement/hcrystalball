import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from fbprophet import Prophet
from hcrystalball.wrappers import ProphetWrapper


@pytest.mark.parametrize(
    "X_with_holidays, extra_holidays",
    [
        (
            "",
            {
                "New year": {"prior_scale": 100},
                "Whit Monday": {"lower_window": -2, "prior_scale": 10},
            },
        ),
        ("", {"New year": {"prior_scale": 100, "lower_window": -1}}),
        ("", None),
    ],
    indirect=["X_with_holidays"],
)
def test_prophet_adjust_holidays(X_with_holidays, extra_holidays):

    prophet = ProphetWrapper(extra_holidays=extra_holidays)
    prophet.model = prophet._init_tsmodel(Prophet)
    X = prophet._adjust_holidays(X_with_holidays)
    holidays = prophet.model.holidays

    assert_frame_equal(X, X_with_holidays.drop(columns="_holiday_DE"))
    assert isinstance(holidays, pd.DataFrame)
    assert "ds" in holidays.columns.tolist()
    assert "holiday" in holidays.columns.tolist()

    assert set(holidays.columns) == set(["ds", "holiday", "lower_window", "upper_window", "prior_scale"])
    if extra_holidays:
        for holiday_name, holiday_params in extra_holidays.items():
            for params_key, params_values in holiday_params.items():
                assert (
                    holidays.loc[holidays["holiday"] == holiday_name + "_DE", params_key].values[0]
                    == params_values
                )


@pytest.fixture(scope="module")
def expected_result_prophet_adjust_holidays_related_features():

    result = {
        "holiday": [
            "New year_BE",
            "Easter Monday_BE",
            "Labour Day_BE",
            "Ascension Thursday_BE",
            "Whit Monday_BE",
            "National Day_BE",
            "Assumption of Mary to Heaven_BE",
            "New year_DE",
            "Good Friday_DE",
            "Easter Monday_DE",
            "Labour Day_DE",
            "Ascension Thursday_DE",
            "Whit Monday_DE",
            "Day of German Unity_DE",
        ],
        "lower_window": [
            -4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -4.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ],
        "upper_window": [
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "prior_scale": [
            100.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            100.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
            10.0,
        ],
        "ds": [
            "2019-01-01",
            "2019-04-22",
            "2019-05-01",
            "2019-05-30",
            "2019-06-10",
            "2019-07-21",
            "2019-08-15",
            "2019-01-01",
            "2019-04-19",
            "2019-04-22",
            "2019-05-01",
            "2019-05-30",
            "2019-06-10",
            "2019-10-03",
        ],
    }

    return pd.DataFrame(result).assign(ds=lambda df: pd.to_datetime(df["ds"]))


@pytest.mark.parametrize(
    "X_with_holidays, extra_holidays, expected_result_prophet_adjust_holidays_related_features",
    [
        (
            "double_holidays",
            {"New year": {"prior_scale": 100, "lower_window": -4}},
            "double_holidays",
        )
    ],
    indirect=[
        "X_with_holidays",
        "expected_result_prophet_adjust_holidays_related_features",
    ],
)
def test_prophet_adjust_holidays_related_features(
    X_with_holidays,
    extra_holidays,
    expected_result_prophet_adjust_holidays_related_features,
):

    prophet = ProphetWrapper(extra_holidays=extra_holidays)
    prophet.model = prophet._init_tsmodel(Prophet)
    prophet._adjust_holidays(X_with_holidays)
    holidays = prophet.model.holidays
    assert_frame_equal(holidays, expected_result_prophet_adjust_holidays_related_features)


@pytest.mark.parametrize(
    "X_y_optional, additional_col",
    [("just_X", None), ("just_X", True), ("", None), ("", True)],
    indirect=["X_y_optional"],
)
def test_prophet_transform_data_to_tsmodel_input_format(X_y_optional, additional_col):

    X, y = X_y_optional
    if additional_col:
        X["additional_col"] = 1
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
        assert "additional_col" in result.columns.tolist()
        col_count += 1
    assert result.shape[1] == col_count
