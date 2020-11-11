import pandas as pd
import numpy as np
import pytest
from hcrystalball.feature_extraction import SeasonalityTransformer
import calendar


weekdays = [f"_{day}" for day in calendar.day_name]
months = [f"_{month}" for month in list(calendar.month_name)[1:]]


@pytest.mark.parametrize(
    "X_start, X_len, weekdays, weeks, months, quarter, year",
    [
        (
            "2019-01-01",
            366,
            weekdays,
            [f"_{i}_week" for i in range(1, 53)],
            months,
            [f"_{i}_quarter" for i in range(1, 5)],
            ["_2019", "_2020"],
        ),
        (
            "2019-01-01",
            365,
            weekdays,
            [f"_{i}_week" for i in range(1, 53)],
            months,
            [f"_{i}_quarter" for i in range(1, 5)],
            ["_2019"],
        ),
        (
            "2019-01-01",
            5,
            set(weekdays).difference(["_Sunday", "_Monday"]),
            ["_1_week"],
            ["_January"],
            ["_1_quarter"],
            ["_2019"],
        ),
    ],
)
def test_seasonality_transformer(X_start, X_len, weekdays, weeks, months, quarter, year):

    X = pd.DataFrame(index=pd.date_range(X_start, periods=X_len, freq="D"))
    y = pd.Series(np.arange(len(X)), name="values", index=X.index)

    freq = pd.infer_freq(X.index)

    df = pd.concat(
        [
            X,
            y,
            SeasonalityTransformer(
                freq=freq,
                month_end=True,
                month_start=True,
                quarter_start=True,
                quarter_end=True,
                year_start=True,
                year_end=True,
            )
            .fit(X, y)
            .transform(X),
        ],
        axis=1,
    )

    periods_starts_ends = [
        "_month_start",
        "_month_end",
        "_quarter_start",
        "_quarter_end",
        "_year_start",
        "_year_end",
    ]
    assert set(weekdays).issubset(df.columns)
    assert set(weeks).issubset(df.columns)
    assert set(months).issubset(df.columns)
    assert set(quarter).issubset(df.columns)
    assert set(year).issubset(df.columns)
    assert set(periods_starts_ends).issubset(df.columns)

    first_row = df.head(1).T
    cols_with_ones = first_row[first_row[first_row.columns[0]] == 1].index

    single_date_cols = (
        SeasonalityTransformer(
            freq=freq,
            month_start=True,
            quarter_start=True,
            year_start=True,
        )
        .fit(X.head(1), y.head(1))
        .transform(X.head(1))
        .columns
    )

    assert set(cols_with_ones) == set(single_date_cols)


def test_seasonality_transformer_ensure_cols():

    X = pd.DataFrame(index=pd.date_range("2019-01-01", periods=10, freq="D"))
    y = pd.Series(np.arange(len(X)), name="values", index=X.index)

    head_t = SeasonalityTransformer(freq="D").fit(X.head(6), y.head(6))
    head_1 = head_t.transform(X.head(6))
    # names of columns in transformer are accessible over get_feature_names after first transform
    assert set(head_1.columns) == set(head_t.get_feature_names())

    head_2 = head_t.transform(X.tail(6))

    assert set(head_1.columns) == set(head_2.columns)

    tail_t = SeasonalityTransformer(freq="D").fit(X.tail(6), y.tail(6))
    tail_1 = tail_t.transform(X.tail(6))

    # missing column was brought back from first transform
    assert ("_Friday" not in tail_1.columns) & ("_Friday" in head_2.columns)
    # extra columns from 2nd and other transforms are clipped away
    assert ("_2_week" in tail_1.columns) & ("_2_week" not in head_2.columns)
