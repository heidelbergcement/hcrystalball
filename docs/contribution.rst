.. _contribution:

Contribution
============

**HCrystalBall** follows same contribution guidelines as scikit-lego_contribution_

.. _scikit-lego_contribution: https://scikit-lego.readthedocs.io/en/latest/contribution.html

Development installation
************************

Our tool of choice for dependencies management is conda mainly due to conflicting requirements of third party packages.
The `environment.yml` from which the environment is build is rather rich and may contain things you might not necessarily need, 
so you can comment out some parts or just take it as an inspiration to build your own.

.. code-block:: bash

   git clone https://github.com/heidelbergcement/hcrystalball
   cd hcrystalball
   conda env create -f environment.yml
   conda activate hcrystalball
   # ensures interactive progress bar will work in example notebooks
   jupyter labextension install @jupyter-widgets/jupyterlab-manager
   python setup.py develop

Building documentation
**********************

Documentation includes examples in form of executable jupyter notebooks and its execution
might take several minutes.

.. code-block:: bash

   python setup.py build_sphinx

.. note::
    
    In case you intend to repeatedly build documentation, executing examples in jupyterlab
    might save you some time, as it runs in parallel. Also, if a notebook has at least 1 output cell,
    sphinx will skip notebooks execution.

If you just want to build sphinx docs without re-executing example notebooks set NBSPHINX_EXECUTE
environment variable to `never`. Default behavior is `auto`.

.. code-block:: bash

   # never execute notebooks
   export NBSPHINX_EXECUTE=never
   # always execute notebooks
   export NBSPHINX_EXECUTE=always
   # execute notebooks that do not have any output cell
   export NBSPHINX_EXECUTE=auto


Running tests
*************

.. code-block:: bash

   # run all tests
   pytest tests
   # run unit tests
   pytest tests/unit
   # run integration tests
   pytest tests/integration