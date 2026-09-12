import os
import sys

repository_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, repository_root)

# Keep the checkout first on PYTHONPATH so autodoc and any explicitly invoked
# documentation helpers import the source tree rather than an older installed
# romtools package.
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = os.pathsep.join(
    path for path in (repository_root, existing_pythonpath) if path
)

project = "ROM Tools and Workflows"
copyright = "2019, National Technology & Engineering Solutions of Sandia, LLC"


def get_version():
    with open(os.path.join(repository_root, "version.txt")) as version_file:
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

# Notebook execution is validated explicitly in docs/validate_examples.py.
# Sphinx is intentionally a rendering-only step so the site build does not
# depend on whether a notebook happened to be saved with outputs.
nb_execution_mode = "off"

# Enable dollar-delimited math ($...$ inline, $$...$$ display) in MyST/myst_nb
# markdown cells. Without this, MyST escapes the dollar signs and MathJax never
# processes the equations.
myst_enable_extensions = ["dollarmath", "amsmath"]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"
exclude_patterns = [
    "_build",
    "**/.ipynb_checkpoints/**",
]

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_nav_level": 1,
    "navigation_depth": 3,
    "collapse_navigation": True,
}
html_sidebars = {
    "**": ["sidebar-nav-bs", "page-toc"],
}
html_css_files = ["custom.css", "demos.css"]
html_js_files = ["ask-repo.js"]
html_title = f"{project} v{release}"
html_static_path = ["_static"]

mathjax3_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"], ["$", "$"]],
        "displayMath": [["\\[", "\\]"], ["$$", "$$"]],
    }
}
