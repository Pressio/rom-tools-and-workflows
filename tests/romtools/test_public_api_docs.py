"""Check the generated reference against the public export boundary."""

import io
from pathlib import Path

import pytest

pytest.importorskip("sphinx")
from sphinx.application import Sphinx


def test_autosummary_generates_public_exports_without_recursive_helpers(tmp_path):
    repository = Path(__file__).resolve().parents[2]
    source = tmp_path / "source"
    source.mkdir()
    configuration = repository / "docs/source/conf.py"
    templates = repository / "docs/source/_templates"
    (source / "conf.py").write_text(
        "import runpy\n"
        f"settings = runpy.run_path({str(configuration)!r})\n"
        "extensions = ['sphinx.ext.autodoc', 'sphinx.ext.autosummary', 'sphinx.ext.napoleon']\n"
        "autosummary_generate = True\n"
        "autosummary_ignore_module_all = settings['autosummary_ignore_module_all']\n"
        "autosummary_context = settings['autosummary_context']\n"
        f"templates_path = [{str(templates)!r}]\n"
        "autodoc_default_options = settings['autodoc_default_options']\n"
    )
    (source / "index.rst").write_text(
        "Public API\n==========\n\n.. autosummary::\n   :toctree: generated\n\n"
        "   romtools.vector_space\n   romtools.vector_space.utils\n"
        "   romtools.hyper_reduction\n   romtools.workflows\n"
        "   romtools.workflows.inverse\n"
        "   romtools.workflows.models\n   romtools.workflows.parameter_spaces\n"
        "   romtools.vector_space.utils.scaler\n"
        "   romtools.vector_space.utils.orthogonalizer\n"
    )
    warnings = io.StringIO()
    application = Sphinx(
        str(source), str(source), str(tmp_path / "out"), str(tmp_path / "doctrees"),
        "dummy", status=io.StringIO(), warning=warnings, freshenv=True,
    )
    application.build(force_all=True)
    assert "duplicate object description" not in warnings.getvalue()
    assert "failed to import" not in warnings.getvalue().lower()
    generated = {path.stem for path in (source / "generated").glob("*.rst")}
    assert "romtools.vector_space.VectorSpaceFromPOD" in generated
    assert "romtools.vector_space.utils.ScalarScaler" in generated
    assert "romtools.hyper_reduction.deim_get_indices" in generated
    assert "romtools.workflows.inverse.run_eki" in generated
    assert "romtools.workflows.run_eki" not in generated
    assert "romtools.vector_space.NoOpScaler" not in generated
    assert "romtools.workflows.inverse.vi_drivers" not in generated
    assert "romtools.hyper_reduction.deim" not in generated
    assert not any(name.endswith((".np", ".Protocol", ".Callable", ".la")) for name in generated)
