import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.pipeline import Pipeline

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
        (
            "series_with_freq_D",
            "LALA",
            None,
            None,
            ValueError,
        ),  # country_code needs to be proper country
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
    X_y_with_freq,
    country_code,
    country_code_column,
    country_code_column_value,
    extected_error,
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
    "country_code, country_code_column, country_code_column_value, exp_col_name",
    [
        ("CZ", None, None, "_holiday_CZ"),
        (None, "holiday_col", "CZ", "_holiday_holiday_col"),
    ],
)
def test_holiday_transformer_transform(
    country_code, country_code_column, country_code_column_value, exp_col_name
):
    expected = {exp_col_name: ["Labour Day", "", "", "", "", "", "", "Liberation Day", "", ""]}
    X = pd.DataFrame(index=pd.date_range(start="2019-05-01", periods=10))
    df_expected = pd.DataFrame(expected, index=X.index)
    if country_code_column:
        X[country_code_column] = country_code_column_value
    df_result = HolidayTransformer(
        country_code=country_code, country_code_column=country_code_column
    ).fit_transform(X)
    assert_frame_equal(df_result, df_expected)


@pytest.mark.parametrize(
    "country_code_first, country_code_column_first, country_code_column_first_value, "
    "country_code_second, country_code_column_second, country_code_column_second_value",
    [
        ("CZ", None, None, "SK", None, None),
        (None, "czech", "CZ", None, "slovak", "SK"),
        ("CZ", None, None, None, "slovak", "SK"),
        (None, "czech", "CZ", "SK", None, None),
    ],
)
def test_two_transformers(
    country_code_first,
    country_code_column_first,
    country_code_column_first_value,
    country_code_second,
    country_code_column_second,
    country_code_column_second_value,
):
    first_suffix = country_code_first or country_code_column_first
    second_suffix = country_code_second or country_code_column_second
    expected = {
        f"_holiday_{first_suffix}": [
            "Labour Day",
            "",
            "",
            "",
            "",
            "",
            "",
            "Liberation Day",
            "",
            "",
        ],
        f"_holiday_{second_suffix}": [
            "Labour Day",
            "",
            "",
            "",
            "",
            "",
            "",
            "Liberation Day",
            "",
            "",
        ],
    }
    X = pd.DataFrame(index=pd.date_range(start="2019-05-01", periods=10))
    df_expected = pd.DataFrame(expected, index=X.index)
    if country_code_column_first:
        X[country_code_column_first] = country_code_column_first_value
    if country_code_column_second:
        X[country_code_column_second] = country_code_column_second_value

    pipeline = Pipeline(
        [
            (
                f"holidays_{first_suffix}",
                HolidayTransformer(
                    country_code_column=country_code_column_first,
                    country_code=country_code_first,
                ),
            ),
            (
                f"holidays_{second_suffix}",
                HolidayTransformer(
                    country_code_column=country_code_column_second,
                    country_code=country_code_second,
                ),
            ),
        ]
    )

    df_result = pipeline.fit_transform(X)
    assert_frame_equal(df_result, df_expected)


@pytest.fixture()
def expected_result_holidays_related_features(request):
    if "without_related_features" in request.param:
        result = {
            "_holiday_DE": [
                "Good Friday",
                "",
                "",
                "Easter Monday",
                "",
                "",
                "",
                "",
                "",
                "",
            ]
        }

    elif "all_related_features" in request.param:
        result = {
            "_holiday_DE": [
                "Good Friday",
                "",
                "",
                "Easter Monday",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "_2_before_holiday_DE": [
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "_2_after_holiday_DE": [
                False,
                True,
                True,
                False,
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            "_bridge_holiday_DE": [
                False,
                True,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }

    elif "features_without_bridge_days" in request.param:
        result = {
            "_holiday_DE": [
                "Good Friday",
                "",
                "",
                "Easter Monday",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "_1_before_holiday_DE": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
            "_1_after_holiday_DE": [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        }

    elif "just_before_holidays_1" in request.param:
        result = {
            "_holiday_DE": [
                "Good Friday",
                "",
                "",
                "Easter Monday",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "_1_before_holiday_DE": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
            ],
        }

    elif "bridge_days_work_just_with_after_and_before_days" in request.param:
        result = {
            "_holiday_DE": [
                "Good Friday",
                "",
                "",
                "Easter Monday",
                "",
                "",
                "",
                "",
                "",
                "",
            ],
            "_1_after_holiday_DE": [
                False,
                True,
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        }

    return pd.DataFrame(result, index=pd.date_range(start="2020-04-10", periods=10))


@pytest.mark.parametrize(
    """country_code,
    days_before,
    days_after,
    bridge_days,
    expected_result_holidays_related_features,
    extected_error""",
    [
        ("DE", 0, 0, False, "without_related_features", None),
        ("DE", 2, 2, True, "all_related_features", None),
        ("DE", 1, 1, False, "features_without_bridge_days", None),
        ("DE", 1, 0, False, "just_before_holidays_1", None),
        (
            "DE",
            0,
            1,
            True,
            "bridge_days_work_just_with_after_and_before_days",
            ValueError,
        ),
    ],
    indirect=["expected_result_holidays_related_features"],
)
def test_holidays_related_features(
    country_code,
    days_before,
    days_after,
    bridge_days,
    expected_result_holidays_related_features,
    extected_error,
):
    X = pd.DataFrame(index=pd.date_range(start="2020-04-10", periods=10))
    if extected_error is None:
        df_result = HolidayTransformer(
            country_code=country_code,
            days_before=days_before,
            days_after=days_after,
            bridge_days=bridge_days,
        ).fit_transform(X)
        assert_frame_equal(df_result, expected_result_holidays_related_features)

    else:
        with pytest.raises(extected_error):
            df_result = HolidayTransformer(
                country_code=country_code,
                days_before=days_before,
                days_after=days_after,
                bridge_days=bridge_days,
            ).fit_transform(X)
