.. _install:

Installation
============

**HCrystalBall** was designed to have soft dependencies on the wrapped libraries 
giving you the opportunity to define your own subset of wrappers that are to be used.

Ideally your application should pin dependencies for wrapped libraries along with 
hcrystalball and other dependencies. 

.. note::

    In case you are having issues with dependencies setup, installing conda environment 
    from `environment.yml` (see :ref:`contribution` for details) will ensure you have evertyhing that is needed.

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