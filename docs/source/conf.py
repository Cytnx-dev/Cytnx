import os,sys
sys.path.append('.')
from link import *


html_theme = 'furo'
#import sphinxbootstrap4theme
#html_theme_path = [sphinxbootstrap4theme.get_path()]

# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'Cytnx'
copyright = '2019-, Kai-Hsin Wu'
author = 'Kai-Hsin Wu'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
# https://sublime-and-sphinx-guide.readthedocs.io/en/latest/references.html
extensions = ['sphinxcontrib.bibtex',\
              'sphinxcontrib.jquery',\
              'sphinx.ext.autosectionlabel',\
              'sphinx_multiversion']
bibtex_bibfiles = ["example/ref.dmrg.bib",\
                   "example/ref.hotrg.bib",\
                   "example/ref.itebd.bib",\
                   "example/ref.idmrg.bib",\
                   "guide/contraction/ref.ncon.bib",\
                   "guide/ref.xlinalg.bib",\
                  ]
extensions.append('sphinx.ext.extlinks')
#extensions.append('sphinx.ext.imgmath')
#extensions.append('ablog')
extensions.append('sphinx.ext.mathjax')

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/versions.html",
        "sidebar/scroll-end.html",
    ],
}
smv_tag_whitelist = r'^v\d+\.\d+\.\d+$'
smv_branch_whitelist = r'main'
smv_remote_whitelist = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_logo = "_static/Iconi.png"

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'pyramid'
#html_theme = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

