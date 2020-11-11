import logging
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from ._split import FinerTimeSplit
from hcrystalball.compose import TSColumnTransformer
from hcrystalball.metrics import get_scorer
from hcrystalball.feature_extraction import HolidayTransformer

logger = logging.getLogger(__name__)


def get_gridsearch(
    frequency,
    horizon=10,
    n_splits=5,
    between_split_lag=None,
    scoring="neg_mean_absolute_error",
    country_code_column=None,
    country_code=None,
    holidays_days_before=0,
    holidays_days_after=0,
    holidays_bridge_days=False,
    sklearn_models=True,
    sklearn_models_optimize_for_horizon=False,
    autosarimax_models=False,
    autoarima_dict=None,
    prophet_models=False,
    tbats_models=False,
    exp_smooth_models=False,
    theta_models=False,
    average_ensembles=False,
    stacking_ensembles=False,
    stacking_ensembles_train_horizon=10,
    stacking_ensembles_train_n_splits=20,
    clip_predictions_lower=None,
    clip_predictions_upper=None,
    exog_cols=None,
):
    """Get grid search object based on selection criteria.

    Parameters
    ----------
    frequency : str
        Frequency of timeseries. Pandas compatible frequncies

    horizon : int
        How many units of frequency (e.g. 4 quarters), should be used to find the best models

    n_splits : int
        How many cross-validation folds should be used in model selection

    between_split_lag : int
        How big lag of observations should cv_splits have
        If kept as None, horizon is used resulting in non-overlaping cv_splits

    scoring : str, callable
        String of sklearn regression metric name, or hcrystalball compatible scorer. For creation
        of hcrystalball compatible scorer use `make_ts_scorer` function.

    country_code_column : str, list
        Column(s) in data, that contain country code in str (e.g. 'DE'). Used in holiday transformer.
        Only one of `country_code_column` or `country_code` can be set.

    country_code : str, list
        Country code(s) in str (e.g. 'DE'). Used in holiday transformer.
        Only one of `country_code_column` or `country_code` can be set.

    holidays_days_before : int
        Number of days before the holiday which will be taken into account
        (i.e. 2 means that new bool column will be created and will be True for 2 days before holidays,
        otherwise False)

    holidays_days_after : int
        Number of days after the holiday which will be taken into account
        (i.e. 2 means that new bool column will be created and will be True for 2 days after holidays,
        otherwise False)

    holidays_bridge_days : bool
        Overlaping `holidays_days_before` and `holidays_days_after` feature which serves for modeling between
        holidays working days

    sklearn_models : bool
        Whether to consider sklearn models

    sklearn_models_optimize_for_horizon: bool
        Whether to add to default sklearn behavior also models, that optimize predictions for each horizon

    autosarimax_models : bool
        Whether to consider auto sarimax models

    autoarima_dict : dict
        Specification of pmdautoarima search space

    prophet_models : bool
        Whether to consider FB prophet models

    exp_smooth_models : bool
        Whether to consider exponential smoothing models

    average_ensembles : bool
        Whether to consider average ensemble models

    stacking_ensembles : bool
        Whether to consider stacking ensemble models

    stacking_ensembles_train_horizon : int
        Which horizon should be used in meta model in stacking ensembles

    stacking_ensembles_train_n_splits : int
        Number of splits used in meta model in stacking ensembles

    clip_predictions_lower : float, int
        Minimal number allowed in the predictions

    clip_predictions_upper : float, int
        Maximal number allowed in the predictions

    exog_cols : list
        List of columns to be used as exogenous variables

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        CV / Model selection configuration
    """
    exog_cols = exog_cols or []
    country_code_columns = (
        [country_code_column] if isinstance(country_code_column, str) else country_code_column
    )
    country_codes = [country_code] if isinstance(country_code, str) else country_code

    # ensures only exogenous columns and country code column will be passed to model if provided
    # and columns names will be stored in TSColumnTransformer
    if exog_cols:
        cols = exog_cols + country_code_columns if country_code_columns else exog_cols
        exog_passthrough = TSColumnTransformer(transformers=[("raw_cols", "passthrough", cols)])
    else:
        exog_passthrough = "passthrough"
    # ensures holiday transformer is added to the pipeline if requested
    if country_codes:
        holiday = Pipeline(
            [
                (
                    f"holiday_{code}",
                    HolidayTransformer(
                        country_code=code,
                        days_before=holidays_days_before,
                        days_after=holidays_days_after,
                        bridge_days=holidays_bridge_days,
                    ),
                )
                for code in country_codes
            ]
        )
    elif country_code_columns:
        holiday = Pipeline(
            [
                (
                    f"holiday_{col}",
                    HolidayTransformer(
                        country_code_column=col,
                        days_before=holidays_days_before,
                        days_after=holidays_days_after,
                        bridge_days=holidays_bridge_days,
                    ),
                )
                for col in country_code_columns
            ]
        )
    else:
        holiday = "passthrough"

    estimator = Pipeline(
        [("exog_passthrough", exog_passthrough), ("holiday", holiday), ("model", "passthrough")]
    )

    cv = FinerTimeSplit(n_splits=n_splits, horizon=horizon, between_split_lag=between_split_lag)

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=[],
        scoring=get_scorer(scoring),
        cv=cv,
        refit=False,
        error_score=np.nan,
    )

    if autosarimax_models:
        # adding autosarimax to param_grid might cause differently found models
        # for different splits and raise inconsistency based errors.
        # sarimax pipeline is added to new grid_search's attribute (`grid_search.autosarimax`)
        # and handled in `hcrystalball.model_seleciton.select_model` function in following way
        # 1. get best model for the data part on last split
        # 2. append this best model to original `param_grid`
        # 3. run full grid search with `param_grid` containing
        #    sarimax model selected from autosarimax in point 1
        from hcrystalball.wrappers import SarimaxWrapper

        if autoarima_dict is None:
            autoarima_dict = {}
        if "error_action" not in autoarima_dict:
            autoarima_dict.update({"error_action": "raise"})

        grid_search.autosarimax = Pipeline(estimator.steps[:-1])
        grid_search.autosarimax.steps.append(
            (
                "model",
                SarimaxWrapper(
                    init_with_autoarima=True,
                    autoarima_dict=autoarima_dict,
                    clip_predictions_lower=clip_predictions_lower,
                    clip_predictions_upper=clip_predictions_upper,
                ),
            )
        )

    if stacking_ensembles or average_ensembles or sklearn_models:
        from sklearn.linear_model import ElasticNet
        from sklearn.ensemble import RandomForestRegressor

        # TODO when scoring time is fixed, add HistGradientBoostingRegressor
        # from sklearn.experimental import enable_hist_gradient_boosting
        # from sklearn.ensemble import HistGradientBoostingRegressor
        from hcrystalball.wrappers import get_sklearn_wrapper
        from hcrystalball.feature_extraction import SeasonalityTransformer

        sklearn_model = get_sklearn_wrapper(
            RandomForestRegressor,
            clip_predictions_lower=clip_predictions_lower,
            clip_predictions_upper=clip_predictions_upper,
        )

        sklearn_model_pipeline = Pipeline(
            [("seasonality", SeasonalityTransformer(auto=True, freq=frequency)), ("model", sklearn_model)]
        )
        # TODO make sure naming here works as expected
        sklearn_model_pipeline.name = f"seasonality_{sklearn_model.name}"

    if sklearn_models:
        classes = [ElasticNet, RandomForestRegressor]
        models = {
            model_class.__name__: get_sklearn_wrapper(
                model_class,
                clip_predictions_lower=clip_predictions_lower,
                clip_predictions_upper=clip_predictions_upper,
            )
            for model_class in classes
        }

        optimize_for_horizon = [False, True] if sklearn_models_optimize_for_horizon else [False]

        grid_search.param_grid.append(
            {
                "model": [sklearn_model_pipeline],
                "model__seasonality__weekly": [True, False],
                "model__model": list(models.values()),
                # TODO change add once HistGradientBoostingRegressor is back
                # "model__model": list(models.values()) + [sklearn_model]
                "model__model__optimize_for_horizon": optimize_for_horizon,
                "model__model__lags": [3, 7, 10, 14],
            }
        )

        grid_search.param_grid.append(
            {
                "model": [sklearn_model_pipeline],
                "model__seasonality__weekly": [True, False],
                "model__model__optimize_for_horizon": optimize_for_horizon,
                "model__model": [sklearn_model],
                "model__model__max_depth": [6],
            }
        )

    if prophet_models:
        from hcrystalball.wrappers import ProphetWrapper

        extra_regressors = [None] if exog_cols is None else [None, exog_cols]

        grid_search.param_grid.append(
            {
                "model": [
                    ProphetWrapper(
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__seasonality_mode": ["multiplicative", "additive"],
                "model__extra_regressors": extra_regressors,
            }
        )

        grid_search.param_grid.append(
            {
                "model": [
                    ProphetWrapper(
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__extra_seasonalities": [
                    [
                        {
                            "name": "quarterly",
                            "period": 90.0625,
                            "fourier_order": 5,
                            "prior_scale": 15.0,
                            "mode": None,
                        }
                    ]
                ],
                "model__extra_regressors": extra_regressors,
            }
        )

    if exp_smooth_models:
        from hcrystalball.wrappers import ExponentialSmoothingWrapper
        from hcrystalball.wrappers import HoltSmoothingWrapper
        from hcrystalball.wrappers import SimpleSmoothingWrapper

        # commented options show non deterministic behavior
        grid_search.param_grid.append(
            {
                "model": [
                    ExponentialSmoothingWrapper(
                        freq=frequency,
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__trend": ["add"],
                "model__seasonal": [None, "add"],
                "model__damped": [True, False],
                "model__fit_params": [
                    {"use_boxcox": True, "use_basinhopping": False},
                    # {'use_boxcox':True, 'use_basinhopping':True},
                    {"use_boxcox": False, "use_basinhopping": False},
                    # {'use_boxcox':False, 'use_basinhopping':True}
                ],
            }
        )

        grid_search.param_grid.append(
            {
                "model": [
                    ExponentialSmoothingWrapper(
                        freq=frequency,
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__trend": ["add"],
                "model__seasonal": ["mul"],
                "model__damped": [True, False],
                "model__fit_params": [
                    {"use_boxcox": False, "use_basinhopping": False},
                    # {'use_boxcox':False, 'use_basinhopping':True}
                ],
            }
        )

        grid_search.param_grid.append(
            {
                "model": [
                    ExponentialSmoothingWrapper(
                        freq=frequency,
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__trend": [None],
                "model__seasonal": [None, "add", "mul"],
                "model__damped": [False],
                "model__fit_params": [
                    {"use_boxcox": False, "use_basinhopping": False},
                    # {'use_boxcox':False, 'use_basinhopping':True}
                ],
            }
        )

        grid_search.param_grid.append(
            {
                "model": [
                    SimpleSmoothingWrapper(
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    ),
                    HoltSmoothingWrapper(
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    ),
                ]
            }
        )

    if theta_models:
        from hcrystalball.wrappers import ThetaWrapper

        grid_search.param_grid.append(
            {
                "model": [
                    ThetaWrapper(
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ]
            }
        )

    if tbats_models:
        from hcrystalball.wrappers import TBATSWrapper

        grid_search.param_grid.append(
            {
                "model": [
                    TBATSWrapper(
                        use_arma_errors=False,
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ]
            }
        )

    if stacking_ensembles:
        from hcrystalball.ensemble import StackingEnsemble
        from hcrystalball.wrappers import ProphetWrapper
        from hcrystalball.wrappers import ThetaWrapper
        from sklearn.ensemble import RandomForestRegressor

        grid_search.param_grid.append(
            {
                "model": [
                    StackingEnsemble(
                        train_n_splits=stacking_ensembles_train_n_splits,
                        train_horizon=stacking_ensembles_train_horizon,
                        meta_model=ElasticNet(),
                        horizons_as_features=True,
                        weekdays_as_features=True,
                        base_learners=[],
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__meta_model": [ElasticNet(), RandomForestRegressor()],
                "model__base_learners": [
                    [
                        ProphetWrapper(
                            clip_predictions_lower=clip_predictions_lower,
                            clip_predictions_upper=clip_predictions_upper,
                        ),
                        sklearn_model_pipeline,
                        ThetaWrapper(
                            clip_predictions_lower=clip_predictions_lower,
                            clip_predictions_upper=clip_predictions_upper,
                        ),
                    ],
                ],
            }
        )
    if average_ensembles:
        from hcrystalball.ensemble import SimpleEnsemble
        from hcrystalball.wrappers import ProphetWrapper
        from hcrystalball.wrappers import ThetaWrapper

        grid_search.param_grid.append(
            {
                "model": [
                    SimpleEnsemble(
                        base_learners=[],
                        clip_predictions_lower=clip_predictions_lower,
                        clip_predictions_upper=clip_predictions_upper,
                    )
                ],
                "model__base_learners": [
                    [
                        ProphetWrapper(
                            clip_predictions_lower=clip_predictions_lower,
                            clip_predictions_upper=clip_predictions_upper,
                        ),
                        sklearn_model_pipeline,
                        ThetaWrapper(
                            clip_predictions_lower=clip_predictions_lower,
                            clip_predictions_upper=clip_predictions_upper,
                        ),
                    ],
                ],
            }
        )

    return grid_search


def add_model_to_gridsearch(model, grid_search):
    """Extends gridsearch with provided model.

    Adds given model or list of models to the gridsearch under 'model' step

    Parameters
    ----------
    model : sklearn compatible model or list of sklearn compatible models
        model(s) to be added to provided grid search

    grid_search : sklearn.model_selection.GridSearchCV
        grid search, that has 'model' step as the last step

    Returns
    -------
    sklearn.model_selection.GridSearchCV
        Grid search enriched with given models
    """
    if isinstance(model, list):
        for mod in model:
            grid_search.param_grid.append({"model": [mod]})
    else:
        grid_search.param_grid.append({"model": [model]})

    return grid_search
