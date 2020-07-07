HCrystal Ball
=============
.. raw:: html

   <div style="display:flex;margin-bottom:20px;">
        <img src="https://raw.githubusercontent.com/heidelbergcement/hcrystalball/master/docs/_static/hcrystal_ball_logo_black.svg" alt="HCrystalBall Logo" style="float:left;width:100px;height:75px;margin-right:15px;">
        <div>
           </br>
           </br>
           <i>
           A library that unifies the API for most commonly used libraries
           and modelling techniques for time-series forecasting in the Python ecosystem.
           </i>
        </div>
   </div>

The library consists of two main parts:

#. :ref:`wrappers` - which bring different 3rd party
   libraries to time series compatible sklearn API
#. :ref:`model_selection` - to enable gridsearch over wrappers, general or custom made transformers
   and add convenient layer over whole process (access to results, plots, storage, ...)

|code| |license|

.. |code| image:: https://img.shields.io/badge/github-code-lightgrey
    :alt: code
    :target: https://github.com/heidelbergcement/hcrystalball
.. |license| image:: https://img.shields.io/github/license/heidelbergcement/hcrystalball
    :alt: license

Quick Installation:
*******************

Install ``HCrystalBall`` from pip

.. code-block:: bash

   pip install hcrystalball

For full installation notes, see :ref:`install`

.. note::

   For questions and contributions, please use |code|

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   glossary
   data_format
   model_selection
   wrappers


.. nbgallery::
    :caption: Examples:
    :name: notebook-gallery
    :glob:

    examples/*

.. toctree::
   :maxdepth: 2
   :caption: API:

   api

.. toctree::
   :caption: Meta:

   contribution
   changelog


Indices and tables:
*******************

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _toctree: http://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html
.. _reStructuredText: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
.. _references: http://www.sphinx-doc.org/en/stable/markup/inline.html
.. _Python domain syntax: http://sphinx-doc.org/domains.html#the-python-domain
.. _Sphinx: http://www.sphinx-doc.org/
.. _Python: http://docs.python.org/
.. _Numpy: http://docs.scipy.org/doc/numpy
.. _SciPy: http://docs.scipy.org/doc/scipy/reference/
.. _matplotlib: https://matplotlib.org/contents.html#
.. _Pandas: http://pandas.pydata.org/pandas-docs/stable
.. _Scikit-Learn: http://scikit-learn.org/stable
.. _autodoc: http://www.sphinx-doc.org/en/stable/ext/autodoc.html
.. _Google style: https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings
.. _NumPy style: https://numpydoc.readthedocs.io/en/latest/format.html
.. _classical style: http://www.sphinx-doc.org/en/stable/domains.html#info-field-lists
.. _Prefect: https://docs.prefect.io/
