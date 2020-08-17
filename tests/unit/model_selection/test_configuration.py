from hcrystalball.model_selection import get_gridsearch
from hcrystalball.model_selection import add_model_to_gridsearch
from hcrystalball.model_selection import FinerTimeSplit
from hcrystalball.feature_extraction import HolidayTransformer
from hcrystalball.compose import TSColumnTransformer
from hcrystalball.wrappers import ProphetWrapper

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pytest


@pytest.mark.parametrize(
    "gridsearch_params, expected_estimator, expected_error",
    [
        (
            {"frequency": "D", "sklearn_models": True, "prophet_models": False},
            {
                "exog_passthrough": str,
                "holiday": str,
                "holiday_step": None,
                "holiday_steps_codes": None,
                "holiday_steps_columns": None,
                "model": str,
            },
            None,
        ),
        (
            {
                "frequency": "D",
                "sklearn_models": True,
                "prophet_models": False,
                "country_code_column": "country",
            },
            {
                "exog_passthrough": str,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": [None],
                "holiday_steps_columns": ["country"],
                "model": str,
            },
            None,
        ),
        (
            {
                "frequency": "D",
                "sklearn_models": True,
                "prophet_models": False,
                "country_code_column": ["czech", "slovak"],
            },
            {
                "exog_passthrough": str,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": [None, None],
                "holiday_steps_columns": ["czech", "slovak"],
                "model": str,
            },
            None,
        ),
        (
            {"frequency": "D", "sklearn_models": True, "prophet_models": False, "country_code": "CZ"},
            {
                "exog_passthrough": str,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": ["CZ"],
                "holiday_steps_columns": [None],
                "model": str,
            },
            None,
        ),
        (
            {"frequency": "D", "sklearn_models": True, "prophet_models": False, "country_code": ["CZ", "SK"]},
            {
                "exog_passthrough": str,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": ["CZ", "SK"],
                "holiday_steps_columns": [None, None],
                "model": str,
            },
            None,
        ),
        (
            {
                "frequency": "D",
                "sklearn_models": True,
                "prophet_models": False,
                "exog_cols": ["raining"],
                "country_code_column": ["czech", "slovak"],
            },
            {
                "exog_passthrough": TSColumnTransformer,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": [None, None],
                "holiday_steps_columns": ["czech", "slovak"],
                "model": str,
            },
            None,
        ),
        (
            {
                "frequency": "D",
                "horizon": 10,
                "n_splits": 4,
                "between_split_lag": 5,
                "scoring": "neg_mean_squared_error",
                "country_code_column": "country",
                "country_code": None,
                "sklearn_models": True,
                "sklearn_models_optimize_for_horizon": True,
                "autosarimax_models": True,
                "autoarima_dict": {"d": 1, "m": 7, "max_p": 2, "max_q": 2},
                "prophet_models": True,
                "tbats_models": True,
                "exp_smooth_models": True,
                "average_ensembles": True,
                "stacking_ensembles": True,
                "stacking_ensembles_train_horizon": 15,
                "stacking_ensembles_train_n_splits": 5,
                "clip_predictions_lower": 0.0,
                "clip_predictions_upper": 1500.0,
                "exog_cols": ["raining"],
            },
            {
                "exog_passthrough": TSColumnTransformer,
                "holiday": Pipeline,
                "holiday_step": HolidayTransformer,
                "holiday_steps_codes": [None],
                "holiday_steps_columns": ["country"],
                "model": str,
            },
            None,
        ),
        ({}, {}, TypeError),
    ],
)
def test_get_gridsearch(gridsearch_params, expected_estimator, expected_error):
    if expected_error is not None:
        with pytest.raises(expected_error):
            get_gridsearch(**gridsearch_params)
    else:
        res = get_gridsearch(**gridsearch_params)

        print(res)
        assert isinstance(res, GridSearchCV)
        assert isinstance(res.cv, FinerTimeSplit)
        assert res.error_score is np.nan
        assert res.refit is False

        assert isinstance(res.estimator["exog_passthrough"], expected_estimator["exog_passthrough"])
        assert isinstance(res.estimator["holiday"], expected_estimator["holiday"])
        if expected_estimator["holiday_step"]:
            assert all(
                [
                    isinstance(holiday_step[1], expected_estimator["holiday_step"])
                    for holiday_step in res.estimator["holiday"].steps
                ]
            )
            assert all(
                [
                    (holiday_step[1].country_code is code) & (holiday_step[1].country_code_column is col)
                    for holiday_step, code, col in zip(
                        res.estimator["holiday"].steps,
                        expected_estimator["holiday_steps_codes"],
                        expected_estimator["holiday_steps_columns"],
                    )
                ]
            )

        assert isinstance(res.estimator["model"], expected_estimator["model"])


def test_add_model_to_gridsearch():
    gs = get_gridsearch(frequency="D", sklearn_models=False)

    model = ProphetWrapper()
    gs = add_model_to_gridsearch(model, gs)

    assert len(gs.param_grid) == 1
    assert str(gs.param_grid[0]["model"][0].get_params()) == str(model.get_params())

    gs = get_gridsearch(frequency="D", sklearn_models=False)

    model = [ProphetWrapper(), ProphetWrapper(clip_predictions_lower=0.0)]

    gs = add_model_to_gridsearch(model, gs)

    assert len(gs.param_grid) == 2
    assert str(gs.param_grid[0]["model"][0].get_params()) == str(model[0].get_params())
    assert str(gs.param_grid[1]["model"][0].get_params()) == str(model[1].get_params())
