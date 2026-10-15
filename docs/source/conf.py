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
# Package export lists define the supported reference surface. Do not recurse
# through every importable implementation module.
autosummary_ignore_module_all = False

# Document each supported object once, using its preferred domain namespace.
# Other export locations link to the same page instead of generating duplicate
# autodoc entries. This also keeps historical aliases visible in the reference.
from importlib import import_module

_public_api_packages = (
    "romtools.vector_space.utils",
    "romtools.vector_space",
    "romtools.composite_vector_space",
    "romtools.hyper_reduction",
    "romtools.linalg",
    "romtools.rom",
    "romtools.workflows.inverse",
    "romtools.workflows.sampling",
    "romtools.workflows.greedy",
    "romtools.workflows.uq",
    "romtools.workflows.models",
    "romtools.workflows.parameter_spaces",
    "romtools.workflows",
    "romtools.vector_space.utils.scaler",
    "romtools.vector_space.utils.orthogonalizer",
)
_public_api_aliases = {}
_canonical_objects = {}
for _package_name in _public_api_packages:
    _package = import_module(_package_name)
    for _name in _package.__all__:
        _object = getattr(_package, _name)
        _export = f"{_package_name}.{_name}"
        _canonical = _canonical_objects.setdefault(id(_object), _export)
        _public_api_aliases[_export] = _canonical

autosummary_context = {"public_api_aliases": _public_api_aliases}

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
    "show_prev_next": False,
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
