import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from hcrystalball.feature_extraction import HolidayTransformer


@pytest.mark.parametrize(
    "X_y_with_freq, country_code, country_code_column, country_code_column_value, extected_error",
    [
        ("series_with_freq_D", "DE", None, None, None),
        ("series_with_freq_D", None, "holiday_col", "DE", None),
        ("series_with_freq_M", "DE", None, None, ValueError),  # not daily freq
        ("series_with_freq_Q", "DE", None, None, ValueError),  # not daily freq
        ("series_with_freq_Y", "DE", None, None, ValueError),  # not daily freq
        (
            "series_with_freq_D",
            None,
            "holiday_colsssss",
            "DE",
            KeyError,
        ),  # there needs to be holiday_col in X
        (
            "series_with_freq_D",
            None,
            None,
            None,
            ValueError,
        ),  # needs to have country_code or country_code_column
        (
            "series_with_freq_D",
            "LALA",
            "LALA",
            None,
            ValueError,
        ),  # cannot have country_code and country_code_column in the same time
        ("series_with_freq_D", "LALA", None, None, ValueError,),  # country_code needs to be proper country
        (
            "series_with_freq_D",
            None,
            "holiday_col",
            "Lala",
            ValueError,
        ),  # country_code needs to be proper country
    ],
    indirect=["X_y_with_freq"],
)
def test_holiday_transformer_inputs(
    X_y_with_freq, country_code, country_code_column, country_code_column_value, extected_error,
):

    X, _ = X_y_with_freq
    if extected_error is not None:
        with pytest.raises(extected_error):
            holiday_transformer = HolidayTransformer(
                country_code=country_code, country_code_column=country_code_column
            )
            if country_code_column:
                X["holiday_col"] = country_code_column_value
            holiday_transformer.fit_transform(X)
    else:
        holiday_transformer = HolidayTransformer(
            country_code=country_code, country_code_column=country_code_column
        )
        if country_code_column:
            X[country_code_column] = country_code_column_value
        holiday_transformer.fit_transform(X)

        if country_code_column:
            assert holiday_transformer.get_params()["country_code"] is None


@pytest.mark.parametrize(
    "country_code, country_code_column, country_code_column_value",
    [("CZ", None, None), (None, "holiday_col", "CZ")],
)
def test_holiday_transformer_transform(country_code, country_code_column, country_code_column_value):
    expected = {"holiday": ["Labour Day", "", "", "", "", "", "", "Liberation Day", "", ""]}
    X = pd.DataFrame(index=pd.date_range(start="2019-05-01", periods=10))
    df_expected = pd.DataFrame(expected, index=X.index)
    if country_code_column:
        X[country_code_column] = country_code_column_value
    df_result = HolidayTransformer(
        country_code=country_code, country_code_column=country_code_column
    ).fit_transform(X)
    assert_frame_equal(df_result, df_expected)
