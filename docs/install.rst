.. _install:

Installation
============

**HCrystalBall** was designed to have soft dependencies on the wrapped libraries
giving you the opportunity to define your own subset of wrappers that are to be used.

Ideally your application should pin dependencies for wrapped libraries along with
hcrystalball and other dependencies.

Install core **HCrystalBall** from pip
***************************************

.. code-block:: bash

   pip install hcrystalball

Install other libraries you want to wrap
*****************************************

.. code-block:: bash

    conda install -c conda-forge statsmodels
    conda install -c conda-forge fbprophet
    conda install -c conda-forge scikit-learn
    conda install -c conda-forge xgboost

    conda install -c alkaline-ml pmdarima

    pip install tbats

For parallel execution
***********************

.. code-block:: bash

    conda install -c conda-forge prefect

Typical Installation
********************

Very often you will want to use more wrappers, than just Sklearn, run examples in jupyterlab,
or execute model selection in parallel. Getting such dependencies to play together nicely
might be cumbersome, so checking `envrionment.yml` might give you faster start.

.. code-block:: bash

    # get dependencies file
    curl -O https://raw.githubusercontent.com/heidelbergcement/hcrystalball/blob/master/environment.yml
    # check comments in environment.yml, keep or remove as requested, than execute
    conda env create -f environment.yml
    conda activate hcrystalball
    # if you want to see progress bar in jupyterlab, execut also
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    # install the library
    pip install hcrystalball