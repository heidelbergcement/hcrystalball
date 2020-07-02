from hcrystalball.model_selection import get_gridsearch
from hcrystalball.model_selection import add_model_to_gridsearch
from hcrystalball.model_selection import FinerTimeSplit
from hcrystalball.feature_extraction import HolidayTransformer
from hcrystalball.compose import TSColumnTransformer
from hcrystalball.wrappers import ProphetWrapper

from sklearn.model_selection import GridSearchCV
import numpy as np
import pytest


@pytest.mark.parametrize(
    "gridsearch_params, expected_estimator, expected_error",
    [
        (
            {"frequency": "D", "sklearn_models": True, "prophet_models": False},
            {"exog_passthrough": str, "holiday": str, "model": str},
            None,
        ),
        (
            {
                "frequency": "D",
                "sklearn_models": True,
                "prophet_models": False,
                "country_code_column": "country",
            },
            {"exog_passthrough": str, "holiday": HolidayTransformer, "model": str},
            None,
        ),
        (
            {"frequency": "D", "sklearn_models": True, "prophet_models": False, "country_code": "DE"},
            {"exog_passthrough": str, "holiday": HolidayTransformer, "model": str},
            None,
        ),
        (
            {
                "frequency": "D",
                "sklearn_models": True,
                "prophet_models": False,
                "exog_cols": ["raining"],
                "country_code_column": "country",
            },
            {"exog_passthrough": TSColumnTransformer, "holiday": HolidayTransformer, "model": str},
            None,
        ),
        ({}, {}, TypeError,),
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
        assert isinstance(res.estimator["model"], expected_estimator["model"])


def test_add_model_to_gridsearch():
    gs = get_gridsearch(frequency="D", prophet_models=False)

    model = ProphetWrapper()
    gs = add_model_to_gridsearch(model, gs)

    assert len(gs.param_grid) == 1
    assert str(gs.param_grid[0]["model"][0].get_params()) == str(model.get_params())

    gs = get_gridsearch(frequency="D", prophet_models=False)

    model = [ProphetWrapper(), ProphetWrapper(clip_predictions_lower=0.0)]

    gs = add_model_to_gridsearch(model, gs)

    assert len(gs.param_grid) == 2
    assert str(gs.param_grid[0]["model"][0].get_params()) == str(model[0].get_params())
    assert str(gs.param_grid[1]["model"][0].get_params()) == str(model[1].get_params())
