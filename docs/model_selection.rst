.. _model_selection:

Model Selection
===============

Throughout different areas, many of us have been dealing with predicting not just one,
but rather multiple time series at a given point in time. Having different regions, countries,
plants, it would have been great to have a tool, that takes state of the art models and for each
partition of the data picks the best model.

This is the main idea, that hcrystalball takes care of for you.

At first, it unifies interface of different libraries under sklearn compatible one using :ref:`wrappers`.
Having such interface enables to build conveniant layer on top using sklearn's model selection pieces and few
custom made additions to make sure, that such task is done in a very pleasant way.

Main interface is `~hcrystalball.model_selection.ModelSelector` class, that aims to cover functionality
around all data partitions. It brings together definition of search space
(`~hcrystalball.model_selection.create_gridsearch`, `~hcrystalball.model_selection.add_model_to_gridsearch`),
runs model selection for each data partition within data (`~hcrystalball.model_selection.select_model`),
provides way how to plot models performance (`~hcrystalball.model_selection.ModelSelector.plot_results`)
and enables load and persistance in unified manner for later reference.

When `~hcrystalball.model_selection.ModelSelector.select_model` is finished, results are stored in
`~hcrystalball.model_selection.ModelSelector.results` - a list of `~hcrystalball.model_selection.ModelSelectorResult` objects,
that are connected with each of the data partition.

Accessing such objects is best done via `~hcrystalball.model_selection.ModelSelector.get_result_for_partition`.

`~hcrystalball.model_selection.ModelSelectorResult` contains majority of information
you would expect to find from model selection process.

.. code-block:: python

    from hcrystalball.model_selection import load_model_selector_result
    msr = load_model_selector_result(partition_hash='fb452abd91f5c3bcb8afa4162c6452c2')
    msr
    ModelSelectorResult
    -------------------
    best_model_name: sklearn
    frequency: D
    horizon: 10

    country_code_column: None

    partition: {'no_partition_label': ''}
    partition_hash: fb452abd91f5c3bcb8afa4162c6452c2

    df_plot: DataFrame of shape (200, 4) suited for plotting cv results with .plot()
    X_train: DataFrame of shape (200, 1) with training feature values
    y_train: DataFrame of shape (200,) with training target values
    cv_results: DataFrame of shape (27, 16) with gridsearch cv info
    best_model_cv_results: Series with gridsearch cv info
    cv_data: DataFrame of shape (20, 29) with models predictions, split and true target values
    best_model_cv_data: DataFrame of shape (20, 3) with model predictions, split and true target values

    model_reprs: Dict of model_hash and model_reprs
    best_model_hash: 53833b776586768405bc49d1944deca7
    best_model: Pipeline(memory=None,
            steps=[('ts_preprocessor',
                    TSColumnTransformer(n_jobs=None, remainder='drop',
                                        sparse_threshold=0.3,
                                        transformer_weights=None,
                                        transformers=[('raw_cols', 'passthrough',
                                                        ['date'])],
                                        verbose=False)),
                    ('model',
                    Pipeline(memory=None,
                            steps=[('sklearn_seasonality',
                                    TSColumnTransformer(n_jobs=None,
                                                        remainder='drop',
                                                        sparse_threshold=0.3,
                                                        transformer...
                                                    importance_type='gain', lags=3,
                                                    learning_rate=0.1,
                                                    max_delta_step=0, max_depth=6,
                                                    min_child_weight=1,
                                                    missing=None, n_estimators=100,
                                                    n_jobs=1, name='sklearn',
                                                    nthread=None,
                                                    objective='reg:squarederror',
                                                    optimize_for_horizon=False,
                                                    random_state=0, reg_alpha=0,
                                                    reg_lambda=1,
                                                    scale_pos_weight=1, seed=None,
                                                    silent=None, subsample=1,
                                                    verbosity=1))],
                            verbose=False))],
            verbose=False)
    -------------------

Plotting
********

On top, `~hcrystalball.model_selection.ModelSelectorResult` provides also convenient plotting functions
(`~hcrystalball.model_selection.ModelSelectorResultplot_result`, `~hcrystalball.model_selection.ModelSelectorResult.plot_error`)
and access to the data behind the plots (`~hcrystalball.model_selection.ModelSelectorResult.df_plot`).

Parallel execution
******************

Model selection itself can also run in parallel using prefect_. For such case ``parallel_over_columns``
must include some categorical columns, that are subset of ``partition_columns``.
Depending on your data, parallel execution might bring unnecessary overhead, so you should treat it carefully.

Predefined parameter grid
*************************

Another built-in method of `~hcrystalball.model_selection.ModelSelector` is `~hcrystalball.model_selection.ModelSelector.create_gridsearch`,
that stores fine-tuned grid to `~hcrystalball.model_selection.ModelSelector.grid_search`
and is by default used in `~hcrystalball.model_selection.ModelSelector.select_model`.

This grid takes care of creation of **holidays** in correct form for each wrapper,
typical set of **seasonality features** like day of the week, ensures, that if passed,
**exogenous columns** are passed correctly and last but not least,
defines set of **models**, that turned out to be useful.

Extending the parameter grid is more than welcomed, as this default might not cover your needs.

How train-test split works
**************************

In the domain of time-series forecasting, we must be extra cautios about using only past information when thinking about predictions.
There are two ways how data can be split in hcrystalball (both are under the hood done by `~hcrystalball.model_selection.FinerTimeSplit`)

Default ``between_split_lag=None`` shifts splits by prediction horizon

.. raw:: html

    <center><img src="https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/docs/_static/FinerTimeSplit.svg" alt="FinerTimeSplit"></center></br>

In case ``between_split_lag`` is defined, it determines the splitting shift as shown below

.. raw:: html

    <center><img src="https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/docs/_static/FinerTimeSplitOverlap.svg" alt="FinerTimeSplitOverlap"></center>

.. _prefect: https://docs.prefect.io/
