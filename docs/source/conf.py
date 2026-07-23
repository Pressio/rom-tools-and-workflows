import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "ROM Tools and Workflows"
copyright = "2019, National Technology & Engineering Solutions of Sandia, LLC"


def get_version():
    with open(os.path.abspath("../../version.txt")) as version_file:
        return version_file.read().strip()


release = get_version()

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_design",
    "myst_nb",
    "jupyter_sphinx",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "show-inheritance": False,
}
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 3,
    "collapse_navigation": True,
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}
html_css_files = ["custom.css"]
html_js_files = ["ask-repo.js"]
html_title = f"{project} v{release}"
html_static_path = ["_static"]

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
    }
}
