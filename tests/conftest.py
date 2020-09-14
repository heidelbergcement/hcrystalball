"""Pytest fixtures."""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from hcrystalball.wrappers import ProphetWrapper
from hcrystalball.wrappers import ExponentialSmoothingWrapper
from hcrystalball.wrappers import TBATSWrapper
from hcrystalball.wrappers import SarimaxWrapper
from hcrystalball.wrappers import get_sklearn_wrapper
from hcrystalball.ensemble import StackingEnsemble, SimpleEnsemble

import pandas._testing as tm

random_state = np.random.RandomState(123)
tm.N = 100  # 100 rows
tm.K = 1  # 1 column


@pytest.fixture(scope="module")
def wrapper_instance(request):

    if request.param == "prophet":
        return ProphetWrapper(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
    elif request.param == "smoothing":
        return ExponentialSmoothingWrapper(trend="add")
    elif request.param == "tbats":
        return TBATSWrapper(use_arma_errors=False, use_box_cox=False)
    elif request.param == "sklearn":
        return get_sklearn_wrapper(LinearRegression, lags=4)
    elif request.param == "sarimax":
        return SarimaxWrapper(order=(1, 1, 0), seasonal_order=(1, 1, 1, 2))
    elif request.param == "stacking_ensemble":
        return StackingEnsemble(
            base_learners=[
                ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                ExponentialSmoothingWrapper(name="smoot_exp2"),
            ],
            meta_model=LinearRegression(),
            horizons_as_features=False,
            weekdays_as_features=False,
        )
    elif request.param == "simple_ensemble":
        return SimpleEnsemble(
            base_learners=[
                ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                ExponentialSmoothingWrapper(name="smoot_exp2"),
            ]
        )


@pytest.fixture(scope="module")
def wrapper_instance_capped(request):
    if request.param.split(";")[0] == "prophet":
        return ProphetWrapper(
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "smoothing":
        return ExponentialSmoothingWrapper(
            trend="add",
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "tbats":
        return TBATSWrapper(
            use_arma_errors=False,
            use_box_cox=False,
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "sklearn":
        return get_sklearn_wrapper(
            LinearRegression,
            lags=4,
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "sarimax":
        return SarimaxWrapper(
            order=(1, 1, 0),
            seasonal_order=(1, 1, 1, 2),
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "stacking_ensemble":
        return StackingEnsemble(
            base_learners=[
                ExponentialSmoothingWrapper(
                    name="smoot_exp1",
                    trend="add",
                    clip_predictions_lower=float(request.param.split(";")[1]),
                    clip_predictions_upper=float(request.param.split(";")[2]),
                ),
                ExponentialSmoothingWrapper(
                    name="smoot_exp2",
                    clip_predictions_lower=float(request.param.split(";")[1]),
                    clip_predictions_upper=float(request.param.split(";")[2]),
                ),
            ],
            meta_model=LinearRegression(),
            horizons_as_features=False,
            weekdays_as_features=False,
            train_n_splits=1,
            train_horizon=10,
            clip_predictions_lower=float(request.param.split(";")[1]),
            clip_predictions_upper=float(request.param.split(";")[2]),
        )
    elif request.param.split(";")[0] == "simple_ensemble":
        return SimpleEnsemble(
            base_learners=[
                ExponentialSmoothingWrapper(
                    name="smoot_exp1",
                    trend="add",
                    clip_predictions_lower=float(request.param.split(";")[1]),
                    clip_predictions_upper=float(request.param.split(";")[2]),
                ),
                ExponentialSmoothingWrapper(
                    name="smoot_exp2",
                    clip_predictions_lower=float(request.param.split(";")[1]),
                    clip_predictions_upper=float(request.param.split(";")[2]),
                ),
            ]
        )


@pytest.fixture(scope="module")
def X_y_linear_trend(request):
    if request.param[-1] not in ("D", "W", "M", "Q", "Y"):
        raise ValueError("Invalid `X_y_with_freq` fixture param.")
    X = pd.DataFrame(
        pd.date_range(start="2019-01-01", periods=100, freq=request.param.split("freq_")[1][0]),
        columns=["date"],
    )

    if "negative" in request.param:
        y = pd.Series(np.linspace(start=80, stop=-19, num=100))
    else:
        y = pd.Series(np.linspace(start=1, stop=100, num=100))

    if "more_cols" in request.param:
        X["trend"] = y + 10
        X["one_hot"] = np.repeat([1, 2, 3, 4], len(X) / 4)

    if "country_col" in request.param:
        X["country"] = "DE"

    if "ndarray" in request.param:
        y = y.values

    if "NaN_y" in request.param:
        y[::9] = np.nan

    if "Inf_y" in request.param:
        y[::15] = np.inf
        y[::16] = -np.inf

    return X.set_index("date"), y


@pytest.fixture(scope="module")
def X_y_optional(request):
    X = pd.DataFrame(index=pd.date_range(start="2019-01-01", periods=300))
    if request.param == "just_X":
        y = None
    else:
        y = np.arange(X.shape[0])
    return X, y


@pytest.fixture(scope="module")
def X_with_holidays(request):
    from hcrystalball.feature_extraction import HolidayTransformer

    X = pd.DataFrame(index=pd.date_range(start="2019-01-01", periods=300))
    holidays = HolidayTransformer(
        country_code="DE", days_before=2, days_after=1, bridge_days=1
    ).fit_transform(X)
    if "double_holidays" in request.param:
        X = X.join(HolidayTransformer(country_code="BE", days_before=0, days_after=2).fit_transform(X))

    return X.join(holidays)


@pytest.fixture(
    scope="module",
    params=[
        "series",
        "series_with_NaN",
        "series_with_Inf",
        "series_with_name",
        "series_with_index_name",
        "dataframe",
        "dataframe_with_NaN",
        "dataframe_with_Inf",
        "dataframe_with_name",
        "dataframe_with_index_name",
        "dataframe_multicolumn",
        "dataframe_integer_index",
        "random_string",
        "emtpy_series",
        "empty_dataframe",
    ],
)
def ts_data(request):
    if "series" in request.param:
        if "empty" in request.param:
            result = pd.Series()
        else:
            result = tm.makeTimeSeries(freq="M")
    elif "dataframe" in request.param:
        if "empty" in request.param:
            result = pd.DataFrame()
        else:
            result = tm.makeTimeDataFrame(freq="M")
            if "multicolumn" in request.param:
                result["dummy_column"] = random_state.random_sample(result.shape[0])

    elif "string" in request.param:
        result = "random_dummy_string"
    else:
        result = None

    if isinstance(result, pd.Series) | isinstance(result, pd.DataFrame):
        if "with_NaN" in request.param:
            result[::2] = np.nan
        if "with_Inf" in request.param:
            result[::3] = np.inf
            result[::6] = -np.inf
        if "with_name" in request.param:
            result.name = "time_series"
        if "with_index_name" in request.param:
            result.index.name = "time_series_index"
        if "integer_index" in request.param:
            result.index = np.arange(result.shape[0], dtype=int)

    return result


@pytest.fixture(scope="module")
def X_y_with_freq(request):
    if request.param[-1] not in ("D", "W", "M", "Q", "Y"):
        raise ValueError("Invalid `X_y_with_freq` fixture param.")

    series = tm.makeTimeSeries(freq=request.param.split("freq_")[1][0])
    X = pd.DataFrame(index=series.index)

    if "series" in request.param:
        y = series
    elif "ndarray" in request.param:
        y = series.values
    else:
        raise ValueError("Invalid `X_y_with_freq` fixture param.")

    if "NaN_y" in request.param:
        y[::9] = np.nan

    if "Inf_y" in request.param:
        y[::10] = np.inf
        y[::11] = -np.inf

    return X, y


@pytest.fixture(scope="module")
def pipeline_instance_model_only(request):

    if request.param == "prophet":
        return Pipeline(
            [
                (
                    "regressor",
                    ProphetWrapper(
                        daily_seasonality=False,
                        weekly_seasonality=False,
                        yearly_seasonality=False,
                    ),
                )
            ]
        )
    elif request.param == "smoothing":
        return Pipeline([("regressor", ExponentialSmoothingWrapper(trend="add"))])

    elif request.param == "tbats":
        return Pipeline([("regressor", TBATSWrapper(use_arma_errors=False, use_box_cox=False))])

    elif request.param == "sklearn":
        return Pipeline([("regressor", get_sklearn_wrapper(LinearRegression, lags=4))])

    elif request.param == "sarimax":
        return Pipeline(
            [
                (
                    "regressor",
                    SarimaxWrapper(order=(1, 1, 0), seasonal_order=(1, 1, 1, 1)),
                )
            ]
        )

    elif request.param == "stacking_ensemble":
        return Pipeline(
            [
                (
                    "regressor",
                    StackingEnsemble(
                        base_learners=[
                            ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                            ExponentialSmoothingWrapper(name="smoot_exp2"),
                        ],
                        meta_model=LinearRegression(),
                    ),
                )
            ]
        )

    elif request.param == "simple_ensemble":
        return Pipeline(
            [
                (
                    "regressor",
                    SimpleEnsemble(
                        base_learners=[
                            ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                            ExponentialSmoothingWrapper(name="smoot_exp2"),
                        ]
                    ),
                )
            ]
        )
    else:
        return None


@pytest.fixture(scope="module")
def pipeline_instance_model_in_pipeline(request):

    if request.param == "prophet":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline(
                        [
                            (
                                "regressor",
                                ProphetWrapper(
                                    daily_seasonality=False,
                                    weekly_seasonality=False,
                                    yearly_seasonality=False,
                                ),
                            )
                        ]
                    ),
                )
            ]
        )

    elif request.param == "smoothing":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline([("regressor", ExponentialSmoothingWrapper(trend="add"))]),
                )
            ]
        )

    elif request.param == "tbats":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline(
                        [
                            (
                                "regressor",
                                TBATSWrapper(use_arma_errors=False, use_box_cox=False),
                            )
                        ]
                    ),
                )
            ]
        )

    elif request.param == "sklearn":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline([("regressor", get_sklearn_wrapper(LinearRegression, lags=4))]),
                )
            ]
        )

    elif request.param == "sarimax":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline(
                        [
                            (
                                "regressor",
                                SarimaxWrapper(order=(1, 1, 0), seasonal_order=(1, 1, 1, 1)),
                            )
                        ]
                    ),
                )
            ]
        )

    elif request.param == "stacking_ensemble":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline(
                        [
                            (
                                "regressor",
                                StackingEnsemble(
                                    base_learners=[
                                        ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                                        ExponentialSmoothingWrapper(name="smoot_exp2"),
                                    ],
                                    meta_model=LinearRegression(),
                                ),
                            )
                        ]
                    ),
                )
            ]
        )

    elif request.param == "simple_ensemble":
        return Pipeline(
            [
                (
                    "model",
                    Pipeline(
                        [
                            (
                                "regressor",
                                SimpleEnsemble(
                                    base_learners=[
                                        ExponentialSmoothingWrapper(name="smoot_exp1", trend="add"),
                                        ExponentialSmoothingWrapper(name="smoot_exp2"),
                                    ]
                                ),
                            )
                        ]
                    ),
                )
            ]
        )
    else:
        return None


@pytest.fixture()
def test_data_raw():
    n_dates = 10
    n_region = 2
    n_plant = 3
    n_product = 4

    dates = ["2018-01-" + str(i) for i in range(1, n_dates + 1)]
    regions = ["region_" + str(i) for i in range(n_region)]
    plants = ["plant_" + str(i) for i in range(n_plant)]
    products = ["product_" + str(i) for i in range(n_product)]

    dfs = []
    for region in regions:
        df_tmp = pd.DataFrame(
            columns=["date", "Region", "Plant", "Product", "Quantity"],
            index=range(len(dates)),
        )
        df_tmp.loc[:, "Region"] = region
        for plant in plants:
            df_tmp.loc[:, "Plant"] = plant
            for product in products:
                df_tmp.loc[:, "date"] = dates
                df_tmp.loc[:, "Product"] = product
                df_tmp.loc[:, "Quantity"] = random_state.random_sample(n_dates)
                dfs.append(df_tmp.copy())

    return pd.concat(dfs).assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")


@pytest.fixture
def train_data(request):
    n_dates = 200
    n_product = 3
    n_regions = 2
    tm.N = n_dates
    tm.K = 1

    df0 = tm.makeTimeDataFrame(freq="D")

    products = ["product_" + str(i) for i in range(n_product)]
    regions = ["region_" + str(i) for i in range(n_regions)]

    dfs = []
    df_tmp = pd.DataFrame(
        columns=["date", "Region", "Product", "Holidays_code", "Quantity"],
        index=range(len(df0.index)),
    )
    for region in regions:
        df_tmp.loc[:, "Region"] = region
        for product in products:
            df_tmp.loc[:, "date"] = df0.index.astype(str).to_list()
            df_tmp.loc[:, "Product"] = product
            df_tmp.loc[:, "Quantity"] = random_state.random_sample(n_dates)
            df_tmp.loc[:, "Holidays_code"] = "NL"

            dfs.append(df_tmp.copy())

    df = pd.concat(dfs).assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")

    if "two_regions" not in request.param:
        return df[df["Region"] == regions[0]].drop(["Region"], axis=1)

    return df


@pytest.fixture()
def grid_search(request):
    from hcrystalball.wrappers import get_sklearn_wrapper
    from hcrystalball.feature_extraction import HolidayTransformer
    from hcrystalball.feature_extraction import SeasonalityTransformer
    from hcrystalball.model_selection import FinerTimeSplit
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.dummy import DummyRegressor
    from hcrystalball.metrics import make_ts_scorer
    from sklearn.metrics import mean_absolute_error

    scoring = make_ts_scorer(mean_absolute_error, greater_is_better=False)

    bad_dummy = get_sklearn_wrapper(
        DummyRegressor, strategy="constant", constant=42, name="bad_dummy", lags=2
    )
    good_dummy = get_sklearn_wrapper(DummyRegressor, strategy="mean", name="good_dummy", lags=2)

    parameters = [
        {"model": [good_dummy]},
        {
            "model": [bad_dummy],
            "model__strategy": ["constant"],
            "model__constant": [42],
        },
    ]

    holiday_model = Pipeline(
        [
            ("holiday", HolidayTransformer(country_code_column="Holidays_code")),
            ("seasonality", SeasonalityTransformer(week_day=True, freq="D")),
            ("model", good_dummy),
        ]
    )
    cv = FinerTimeSplit(n_splits=2, horizon=5)
    grid_search = GridSearchCV(holiday_model, parameters, cv=cv, scoring=scoring)

    return grid_search


@pytest.fixture()
def unprepared_data():
    increment = pd.DataFrame(
        {
            "date": ["2019-05-01", "2019-05-01", "2019-05-01", "2019-05-02"],
            "cem_type": ["CEM 52,5 R-ft", "CEM 52,5 R-ft", "CEM 52,5 N", "CEM 52,5 R"],
            "dummy": ["this", "is", "dummy", "data"],
            "delivery_quantity": [23.0, 27.0, 25.0, 5.0],
        }
    )
    return increment.assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")


@pytest.fixture
def prepared_data(request):
    if "without_logical_partition" in request.param:

        increment = pd.DataFrame(
            {
                "date": ["2019-05-01", "2019-05-02"],
                "target": [75.0, 5.0],
                "cem_type": ["CEM 52,5 N", "CEM 52,5 R"],
                "dummy": ["dummy", "data"],
            }
        )

    else:
        increment = pd.DataFrame(
            {
                "date": ["2019-05-01", "2019-05-02", "2019-05-01"],
                "cem_type": ["CEM 52,5 N", "CEM 52,5 R", "CEM 52,5 R-ft"],
                "dummy": ["dummy", "data", "is"],
                "target": [25, 5.0, 50.0],
            }
        )
        increment = increment.astype({"cem_type": "category"})

    return increment.assign(date=lambda x: pd.to_datetime(x["date"])).set_index("date")
