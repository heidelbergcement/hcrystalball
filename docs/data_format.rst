.. _data_format:

Data Format
===========

In hcrystalball the wrapper models and model selection functions follow the scikit-learn API,
which allows using the scikit-learn grid search, metrics and other tools.

This page describes the data format used for time series, feature and target data in hcrystalball.

Model wrappers
**************

Requested data format for wrappers semantically follows scikit-learn's convention - ``X`` is a feature matrix and ``y`` stands for the target vector.
Along with that we enforce following rules:

- ``X_train`` must be a `pandas.DataFrame` that contains an index of type `pandas.DatetimeIndex`.
- ``y_train`` can be either a `pandas.Series` or `numpy.ndarray` with same length as ``X_train``.
- ``X_test`` shares same format as ``X_train``, while it's length determines for how many steps ahead the wrapper will be predicting.
- ``y_pred`` is always a `pandas.Series` with a `pandas.DatetimeIndex` named after wrapper's name for convenient plotting and pipelining

Following example creates dummy data in the right format with `~hcrystalball.utils.generate_tsdata`
and uses it in `~hcrystalball.wrappers.ProphetWrapper`.

.. code-block:: python

    from hcrystalball.utils import generate_tsdata
    from hcrystalball.wrappers import ProphetWrapper

    X, y = generate_tsdata(n_dates=365*2)
    X_train, y_train, X_test, y_test = X[:-10], y[:-10], X[-10:], y[-10:]

    model = ProphetWrapper()
    y_pred = model.fit(X_train,y_train).predict(X_test)


.. code-block:: python

    X.head()
    Empty DataFrame
    Columns: []
    Index: [2017-01-01 00:00:00, 2017-01-02 00:00:00, 2017-01-03 00:00:00, 2017-01-04 00:00:00, 2017-01-05 00:00:00]

    [730 rows x 0 columns]
    y
    2017-01-01    4.154750
    2017-01-02    6.361124
    2017-01-03    7.676185
    2017-01-04    8.447134
    2017-01-05    8.638612
                    ...
    2018-12-27    5.824521
    2018-12-28    5.359175
    2018-12-29    5.093221
    2018-12-30    6.148416
    2018-12-31    8.176576
    Name: target, Length: 730, dtype: float64

.. note::

    In case you are fitting your model on whole data and you use some exogenous variables
    (e.g. columns with weather forecast), these columns must also be present in ``X_test``.
    In this example it would mean, that you need to provide weather forecast for each
    step ahead along the with the date index.

Model selection
***************

More general model selection interface expects single `pandas.DataFrame`, that must contain at minimum
an index of type `pandas.DatetimeIndex` and a numeric target column. In this case the target is ``Quantity``, index can have a name,
but it is never used

Other columns:

- columns serving to **partition** data (``Region``, ``Plant``, ``Product``),
  that will effectively cut the original data to single time series (similar to X,y format of the wrapper layer)
- **exogenous columns** that add extra information to the autoregressive nature of target prediction (``Raining``)
- a column with ISO code of country/region (``Country``), that is later used to create **holidays** as additional features

This time, dummy data is created with `~hcrystalball.utils.generate_multiple_tsdata` and analysed with `~hcrystalball.model_selection.ModelSelector`.

.. code-block:: python

    from hcrystalball.utils import generate_multiple_tsdata
    from hcrystalball.model_selection import ModelSelector

    df = generate_multiple_tsdata(n_dates=200, n_regions=2, n_plants=2, n_products=2)

    ms = ModelSelector(horizon=10, frequency="D", country_code_column="Country")
    ms.create_gridsearch(n_splits=2, sklearn_models=True, prophet_models=False, exog_cols=["Raining"])
    ms.select_model(df=df, target_col_name="Quantity", partition_columns=["Region", "Plant", "Product"])

.. code-block:: python

    df.head()
                Region    Plant    Product   Country  Raining   Quantity
    Date
    2018-01-01  region_0  plant_0  product_0      DE    False   5.551729
    2018-01-02  region_0  plant_0  product_0      DE    False   8.026498
    2018-01-03  region_0  plant_0  product_0      DE     True   9.120487
    2018-01-04  region_0  plant_0  product_0      DE     True  10.601816
    2018-01-05  region_0  plant_0  product_0      DE     True  10.833782
