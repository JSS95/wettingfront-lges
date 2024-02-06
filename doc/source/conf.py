# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess

import yaml

from wettingfront_lges import get_sample_path

os.environ["WETTINGFRONT_SAMPLES"] = get_sample_path()

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "WettingFront-LGES"
copyright = "2024, Jisoo Song"
author = "Jisoo Song"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx.ext.intersphinx",
    "sphinx_tabs.tabs",
    "matplotlib.sphinxext.plot_directive",
]

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore

autodoc_typehints = "description"

autoapi_dirs = ["../../src"]
autoapi_template_dir = "_templates/autoapi"

intersphinx_mapping = {
    "python": ("http://docs.python.org/", None),
    "pip": ("https://pip.pypa.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "cattrs": ("https://cattrs.readthedocs.io/en/latest/", None),
    "mypy": ("https://mypy.readthedocs.io/en/stable/", None),
    "PIL": ("https://pillow.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_title = "WettingFront-LGES"
html_static_path = []  # type: ignore

plot_html_show_formats = False
plot_html_show_source_link = False

# -- Custom scripts ----------------------------------------------------------

# Tutorial files

with open("example.yml", "r") as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

subprocess.call(
    [
        "wettingfront",
        "analyze",
        "example.yml",
    ],
)
