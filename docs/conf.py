"""Sphinx docs build configuration file."""
import os
import inspect
import hcrystalball

__location__ = os.path.join(os.getcwd(), os.path.dirname(inspect.getfile(inspect.currentframe())))

extensions = [
    "sphinx_automodapi.automodapi",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.ifconfig",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "nbsphinx",
    "sphinx_gallery.load_style",
]
numpydoc_show_class_members = False

templates_path = ["_templates"]

source_suffix = [".rst"]

default_role = "py:obj"

master_doc = "index"

# General information about the project.
project = "hcrystalball"
copyright = "2020, Pavel, Michal, Jan, Attila and others[tm]"

version = hcrystalball.__version__
release = version

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    # 'sidebar_width': '300px',
    # 'page_width': '1000px'
}

html_logo = "_static/hcrystal_ball_logo_white.svg"

html_static_path = ["_static"]

html_context = {
    "css_files": ["_static/theme_overrides.css"],
}

htmlhelp_basename = "hcrystalball-doc"

intersphinx_mapping = {
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
    "python": ("https://docs.python.org/3/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "statsmodels": ("https://www.statsmodels.org/dev", None),
    "pmdarima": ("http://alkaline-ml.com/pmdarima/", None),
}

autosummary_generate = True
automodsumm_inherited_members = True

# Handling notebook execution
nbsphinx_execute = os.getenv("NBSPHINX_EXECUTE", "auto")
nbsphinx_kernel_name = "python3"
