![CI (conda)](https://github.com/heidelbergcement/hcrystalball/workflows/CI%20(conda)/badge.svg)
![CD (PyPI)](https://github.com/heidelbergcement/hcrystalball/workflows/CD%20(PyPI)/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/hcrystalball/badge/?version=latest)](https://hcrystalball.readthedocs.io/en/latest/?badge=latest)
![License](https://img.shields.io/github/license/heidelbergcement/hcrystalball)
![Contributors](https://img.shields.io/github/contributors/heidelbergcement/hcrystalball)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

TODO
[shields]
 - downloads
 - pypi version
 - conda forge version
 - python versions

# HCrystal Ball

<div>
<a href="https://hcrystalball.readthedocs.io/en/latest/"><img src="docs/_static/hcrystal_ball_logo_green.png" width="150px" align="left" /></a>
<i><br>A time series library that unifies the API for most commonly <br> 
used libraries and modelling techniques for time-series <br> 
forecasting in the Python ecosystem.</i>
</div>
<br><br><br>

**HCrystal Ball** consists of two main parts:

* **Wrappers** - which bring different 3rd party 
   libraries to time series compatible sklearn API
* **Model Selection** - to enable gridsearch over wrappers, general or custom made transformers
   and add convenient layer over whole process (access to results, plots, storage, ...)

## Documentation
See examples, tutorials, contribution, API and more on the documentation [site](https://hcrystalball.readthedocs.io/en/latest) or browse [example notebooks](https://github.com/heidelbergcement/hcrystalball/tree/master/docs/examples) directly.

## Core Installation

If you want really minimal installation, you can install from pip

```bash
pip install hcrystalball
```

## Typical Installation

Very often you will want to use more wrappers, than just Sklearn, run examples in jupyterlab, or execute model selection in parallel. Getting such dependencies to play together nicely might be cumbersome, so checking `envrionment.yml` might give you faster start.

```bash
# get dependencies file
curl -O https://raw.githubusercontent.com/heidelbergcement/hcrystalball/blob/master/environment.yml
# check comments in environment.yml, keep or remove as requested, than execute
conda env create -f environment.yml
conda activate hcrystalball
# if you want to see progress bar in jupyterlab, execut also
jupyter labextension install @jupyter-widgets/jupyterlab-manager
# install the library
pip install hcrystalball
```

## Development Installation:

To have everything in place including docs build or executing tests, execute following code

```bash
git clone https://github.com/heidelbergcement/hcrystalball
cd hcrystalball
conda env create -f environment.yml
conda activate hcrystalball
# ensures interactive progress bar will work in example notebooks
jupyter labextension install @jupyter-widgets/jupyterlab-manager
python setup.py develop
```

## Example Usage
### Wrappers
```python
from hcrystalball.utils import generate_tsdata
from hcrystalball.wrappers import ProphetWrapper

X, y = generate_tsdata(n_dates=365*2)
X_train, y_train, X_test, y_test = X[:-10], y[:-10], X[-10:], y[-10:]

model = ProphetWrapper()
y_pred = model.fit(X_train, y_train).predict(X_test)
y_pred
            prophet
2018-12-22  6.066999
2018-12-23  6.050076
2018-12-24  6.105620
2018-12-25  6.141953
2018-12-26  6.150229
2018-12-27  6.163615
2018-12-28  6.147420
2018-12-29  6.048633
2018-12-30  6.031711
2018-12-31  6.087255
```

### Model Selection

```python
from hcrystalball.utils import generate_multiple_tsdata
from hcrystalball.model_selection import ModelSelector

df = generate_multiple_tsdata(n_dates=200, 
                              n_regions=1, 
                              n_plants=1, 
                              n_products=2,
                              )

ms = ModelSelector(horizon=10, 
                   frequency="D", 
                   country_code_column="Country",
                   )

ms.create_gridsearch(n_splits=2, 
                     sklearn_models=True, 
                     prophet_models=False, 
                     exog_cols=["Raining"],
                     )

ms.select_model(df=df, 
                target_col_name="Quantity", 
                partition_columns=["Region", "Plant", "Product"],
                )

# Model Selector is updated with results
ms

ModelSelector
-------------
  frequency: D
  horizon: 10
  country_code_column: Country
  results: List of 2 ModelSelectorResults
  paritions: List of 2 partitions
     {'Plant': 'plant_0', 'Product': 'product_0', 'Region': 'region_0'}
     {'Plant': 'plant_0', 'Product': 'product_1', 'Region': 'region_0'}
-------------

# Accessing result for 1 partition showcases rich representation
ms.results[0]

ModelSelectorResult
-------------------
  best_model_name: sklearn
  frequency: D
  horizon: 10

  country_code_column: None

  partition: {'Plant': 'plant_0', 'Product': 'product_0', 'Region': 'region_0'}
  partition_hash: 094a99e51ce41bad546788ddb8380ac1

  df_plot: DataFrame of shape (200, 6) suited for plotting cv results with .plot()
  X_train: DataFrame of shape (200, 2) with training feature values
  y_train: DataFrame of shape (200,) with training target values
  cv_results: DataFrame of shape (18, 16) with gridsearch cv info
  best_model_cv_results: Series with gridsearch cv info
  cv_data: DataFrame of shape (20, 20) with models predictions, split and true target values
  best_model_cv_data: DataFrame of shape (20, 3) with model predictions, split and true target values

  model_reprs: Dict of model_hash and model_reprs
  best_model_hash: cbc68abad45e02bec6b2de157bc8c396
  best_model: Pipeline(memory=None,
         steps=[('exog_passthrough',
                 TSColumnTransformer(n_jobs=None, remainder='drop',
                                     sparse_threshold=0.3,
                                     transformer_weights=None,
                                     transformers=[('raw_cols', 'passthrough',
                                                    ['Raining'])],
                                     verbose=False)),
                ('holiday', 'passthrough'),
                ('model',
                 Pipeline(memory=None,
                          steps=[('seasonality',
                                  SeasonalityTransformer(auto=True, freq='D',
                                                         monthly=None,
                                                         quar...
                                  SklearnWrapper(alpha=1.0,
                                                 clip_predictions_lower=None,
                                                 clip_predictions_upper=None,
                                                 copy_X=True,
                                                 fit_intercept=True,
                                                 fit_params=None, l1_ratio=0.5,
                                                 lags=14, max_iter=1000,
                                                 name='sklearn',
                                                 normalize=False,
                                                 optimize_for_horizon=False,
                                                 positive=False,
                                                 precompute=False,
                                                 random_state=None,
                                                 selection='cyclic', tol=0.0001,
                                                 warm_start=False))],
                          verbose=False))],
         verbose=False)
-------------------
```
