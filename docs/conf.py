# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys
import tomllib
import better_apidoc

conf_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.dirname(conf_path)
lib_path = os.path.join(project_path, 'open_qbench')
sys.path.insert(0, project_path)


def run_apidoc(app):
    """Generate API documentation"""
    better_apidoc.APP = app
    better_apidoc.main([
        'better-apidoc',
        '-t',
        os.path.join('.', '_templates'),
        '--force',
        '--separate',
        '-o',
        os.path.join('.', 'API'),
        lib_path,  # Generate docs only for quantum_launcher/
    ])


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# Grab info from pyproject.toml

with open(os.path.join(project_path, 'pyproject.toml'), 'rb') as f:
    parsed_pyproject = tomllib.load(f)

name = parsed_pyproject['project']['name'].replace('_', ' ').title()
version = parsed_pyproject['project']['version']

project = name
copyright = '2025, psnc-qcg'
author = 'psnc-qcg'
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

suppress_warnings = [
    'myst.header',  # ipynb linter
    'docutils'  # rst linter
]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_copybutton',
    'myst_nb',
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
]

nb_execution_mode = "off"

viewcode_line_numbers = True

autodoc_typehints = "both"
autodoc_typehints_description_target = "documented"

napoleon_numpy_docstring = False

templates_path = ['_templates']
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'tests/*',
    '*.md',
    'API/modules.rst'
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']
html_css_files = ['css/custom.css']

html_theme = 'sphinx_book_theme'
html_logo = '_static/qbench_logo.png'
html_favicon = "_static/logo.svg"

html_show_sourcelink = False  # Disable option to show .rst source


def setup(app):
    app.connect('builder-inited', run_apidoc)
    app.add_css_file('css/custom.css')
