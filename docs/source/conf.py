# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
sys.path.insert(0, os.path.abspath('../../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'py_libraries'
copyright = '2026, Esteban Thevenon'
author = 'Esteban Thevenon'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',         # reads docstrings
    'sphinx.ext.napoleon',        # NumPy/Google style docstrings
    'sphinx.ext.viewcode',        # adds [source] links
    'sphinx.ext.autosummary',     # generates summary tables
    'sphinx_autodoc_typehints',   # renders type hints
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
autodoc_typehints = 'description'
napoleon_numpy_docstring  = True
napoleon_google_docstring = True
autosummary_generate = True

autodoc_mock_imports = [
    "pyautogui",
]