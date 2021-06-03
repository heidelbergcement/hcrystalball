.. _wrappers:

Wrappers
========

It is not that exceptional, that different time-series libraries have different interaface towards
data, way how their models are fitted, how the prediction is made, which functionality happens at
models initialization and which happens somewhere along the way. Each library has specific needs for
handling holidays, seasonalities or exogenous variables.

If you are facing a task of using best models, that python eco-system provides, you would most probably
bring different notebooks with specific data preparation for each of the library you plan to use.

Later for each of the use-case, or data partition, you would need to adapt your code to fit the library's needs.

Also, searching for optimal model parameters and data preprocessing steps might become tedious task in such setup.

Main building block of hcrystalball, that further enables such large scale cross validation, is a layer of wrappers,
that bring models from prophet, exponential smoothing (from statsmodels), sarimax (from pmdarima), tbats
and any sklearn compatible regressor to time-series compatible sklearn API compliant nature.

Usage of such wrappers is for people who are familiar with sklearn straight forward

Most wrappers
*****************************

.. code-block:: python

    # prophet, pmdarima, statsmodels, tbats
    from hcrystalball.wrappers import SarimaxWrapper
    # from hcrystalball.wrappers import ProphetWrapper
    # from hcrystalball.wrappers import ExponentialSmoothingWrapper
    # from hcrystalball.wrappers import SimpleSmoothingWrapper
    # from hcrystalball.wrappers import HoltSmoothingWrapper
    # from hcrystalball.wrappers import TBATSWrapper
    # from hcrystalball.wrappers import BATSWrapper

    model = SarimaxWrapper(order=(4, 1, 5), seasonal_order=(0,0,0,0))
    model.fit(X[:-10], y[:-10])
    model.predict(X[-10:]

In majority of the cases above, you simply import the wrapper and have a combination of helpful
custom made parameters and wrapped model parameters.

For example SarimaxWrapper allows you to initialize the model with automatic finding of Sarimax
parameters with ``init_with_autoarima``, but you might also directly instantiate a SarimaxWrapper
with custom order - parameter, that wrapped model is exposing.


Sklearn compatible regressors
*****************************

.. code-block:: python

    from hcrystalball.wrappers import get_sklearn_wrapper
    from sklearn.linear_model import LinearRegression

    # sklearn compatible regressors
    model = get_sklearn_wrapper(LinearRegression)
    model.fit(X[:-10], y[:-10])
    model.predict(X[-10:]

You might have mentioned, that there is a special factory function ``get_sklearn_wrapper`` that needs to be used in case
of sklearn compatible regressors. Reason behind is that this time when importing SklearnWrapper we wouldn't
know, which model is to be wrapped (as oppose to the other wrappers, where we have exact classes that we wrap).
