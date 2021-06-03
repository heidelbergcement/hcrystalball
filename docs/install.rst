.. _install:

Installation
============

**HCrystalBall** was designed to have soft dependencies on the wrapped libraries
giving you the opportunity to define your own subset of wrappers that are to be used.

Ideally your application should pin dependencies for wrapped libraries along with
hcrystalball and other dependencies.

Install core **HCrystalBall** from pip or from conda-forge
**********************************************************

.. code-block:: bash

    pip install hcrystalball

.. code-block:: bash

    conda install -c conda-forge hcrystalball

Install other libraries you want to wrap
*****************************************

.. code-block:: bash

    conda install -c conda-forge statsmodels
    conda install -c conda-forge prophet
    conda install -c conda-forge scikit-learn

    pip install pmdarima
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
