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

Creating new release
********************
Publishing new package version to PyPI and conda-forge is done withing continuous deployment, that is setup on the creation of new release.
Even that some steps are automated, make sure to go through following checklist to ensure the best outcome of a new release.

  #. Fork the original repository
  #. Make changes to the code within forked repository
  #. Create pull request against the original repository
  #. Repeat until all things that you wanted in the release are merged to the master
  #. If CHANGELOG.rst was not updated, make sure you update it with a new version
  #. Create `new release <https://help.github.com/en/enterprise/2.13/user/articles/creating-releases>`_ from master with new tag (e.g. v0.2.1).
     Keep the description blank to have single source of truth in CHANGELOG.rst
  #. Check the results of workflows in `GitHub Actions <https://github.com/heidelbergcement/hcrystalball/actions>`_
  #. Check the new release is available on `PyPI <https://pypi.org/project/hcrystalball/>`_
  #. After 1 hour check that conda-forge bot published the new release on `conda-forge <https://conda-forge.org/feedstocks/>`_
  #. Take some rest with your favorite drink