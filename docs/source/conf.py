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
#_version = 'v0.9.7'
#version = 'v0.5.5a'

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



html_theme_options = {
    # Navbar style.
    # Values: 'fixed-top', 'full' (Default: 'fixed-top')
    'navbar_style' : 'fixed-top',

    # Navbar link color modifier class.
    # Values: 'dark', 'light' (Default: 'dark')
    'navbar_color_class' : 'dark',

    # Navbar background color class.
    # Values: 'inverse', 'primary', 'faded', 'success',
    #         'info', 'warning', 'danger' (Default: 'inverse')
    #'navbar_bg_class' : 'inverse',

    # Show global TOC in navbar.
    # To display up to 4 tier in the drop-down menu.
    # Values: True, False (Default: True)
    'navbar_show_pages' : True,

    # Link name for global TOC in navbar.
    # (Default: 'Pages')
    'navbar_pages_title' : 'Pages',

    # Specify a list of menu in navbar.
    # Tuples forms:
    #  ('Name', 'external url or path of pages in the document', boolean)
    # Third argument:
    # True indicates an external link.
    # False indicates path of pages in the document.
    'navbar_links' : [
         ('API Doc', 'https://kaihsinwu.gitlab.io/cytnx_api', True),
         ("Github", "https://github.com/Cytnx-dev/Cytnx", True),
    ],

    # Total width(%) of the document and the sidebar.
    # (Default: 80%)
    'main_width' : '80%',

    # Render sidebar.
    # Values: True, False (Default: True)
    'show_sidebar' : True,

    # Render sidebar in the right of the document.
    # Values：True, False (Default: False)
    'sidebar_right': False,

    # Fix sidebar.
    # Values: True, False (Default: True)
    'sidebar_fixed': False,

    # Html table header class.
    # Values: 'inverse', 'light' (Deafult: 'inverse')
    'table_thead_class' : 'inverse',

}
"""
# -- Breathe configuration -------------------------------------------------

breathe_projects = {
	"Cyc": "/home/kaihsinwu/Dropbox/Cytnx/docs/xml/"
}

breathe_default_project = "Cyc"
breathe_default_members = ('members', 'undoc-members')
#breathe_default_members = ('members')
"""
## ablog
#import ablog
#templates_path.append(ablog.get_html_templates_path())
#disqus_shortname='kaihsinwu'


