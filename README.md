[![CI](https://github.com/heidelbergcement/hcrystalball/workflows/CI/badge.svg)](https://github.com/heidelbergcement/hcrystalball/actions/?query=workflow%3ACI)
[![CD](https://github.com/heidelbergcement/hcrystalball/workflows/CD/badge.svg)](https://github.com/heidelbergcement/hcrystalball/actions?query=workflow%3ACD)
[![Documentation Status](https://readthedocs.org/projects/hcrystalball/badge/?version=latest)](https://hcrystalball.readthedocs.io/en/latest/?badge=latest)
[![PyPI Version](https://img.shields.io/pypi/v/hcrystalball)](https://pypi.org/project/hcrystalball/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/hcrystalball)](https://anaconda.org/conda-forge/hcrystalball)
[![Code Coverage](https://codecov.io/gh/heidelbergcement/hcrystalball/branch/master/graph/badge.svg)](https://codecov.io/gh/heidelbergcement/hcrystalball)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/heidelbergcement/hcrystalball/master?filepath=docs/examples/)
[![License](https://img.shields.io/github/license/heidelbergcement/hcrystalball)](https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/LICENSE.txt)
[![Contributors](https://img.shields.io/github/contributors/heidelbergcement/hcrystalball)](https://github.com/heidelbergcement/hcrystalball/graphs/contributors)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# HCrystal Ball

<div>
    <a href="https://hcrystalball.readthedocs.io/en/latest/">
        <img src="https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/docs/_static/hcrystal_ball_logo_green.png"    width="150px" align="left" /></a>
    <i><br>A library that unifies the API for most commonly <br>
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
See examples, tutorials, contribution, API and more on the documentation [site](https://hcrystalball.readthedocs.io/en/latest) try notebooks on [binder](https://mybinder.org/v2/gh/heidelbergcement/hcrystalball/master) or browse example notebooks in [docs/examples](https://github.com/heidelbergcement/hcrystalball/tree/master/docs/examples) directly.

## Core Installation

If you want really minimal installation, you can install from pip or from conda-forge

```bash
pip install hcrystalball
```

```bash
conda install -c conda-forge hcrystalball
```

## Typical Installation

Very often you will want to use more wrappers, than just Sklearn, run examples in jupyterlab, or execute model selection in parallel. Getting such dependencies to play together nicely might be cumbersome, so checking `envrionment.yml` might give you faster start.

```bash
# get dependencies file, e.g. using curl
curl -O https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/environment.yml
# check comments in environment.yml, keep or remove as requested, than create environment using
conda env create -f environment.yml
# activate the environment
conda activate hcrystalball
# if you want to see progress bar in jupyterlab, execute also
jupyter labextension install @jupyter-widgets/jupyterlab-manager
# install the library from pip
pip install hcrystalball
# or from conda
conda install -c conda-forge hcrystalball
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
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 6]

from hcrystalball.utils import get_sales_data
from hcrystalball.model_selection import ModelSelector

df = get_sales_data(n_dates=200,
                    n_assortments=1,
                    n_states=2,
                    n_stores=2)

ms = ModelSelector(horizon=10,
                   frequency="D",
                   country_code_column="HolidayCode",
                   )

ms.create_gridsearch(n_splits=2,
                     sklearn_models=True,
                     prophet_models=False,
                     exog_cols=["Open","Promo","SchoolHoliday","Promo2"],
                     )

ms.select_model(df=df,
                target_col_name="Sales",
                partition_columns=["Assortment", "State","Store"],
                )

ms.plot_results(plot_from="2015-06-01",
                partitions=[{"Assortment":"a","State":"NW","Store":335}]
               )
```

<img src="https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/docs/_static/forecast.png" width="100%" align="left"/>
