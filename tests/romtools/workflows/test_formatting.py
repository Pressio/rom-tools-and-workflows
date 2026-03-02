from pathlib import Path
import pytest

from romtools.workflows.formatting import format_text, format_file, format_files, _parse_params_file


def test_format_text_replaces_values():
    text = "alpha={RTVAR:alpha}, beta={RTVAR:beta}"
    variables = {"alpha": 1.5, "beta": "two"}
    assert format_text(text, variables) == "alpha=1.5, beta=two"


def test_format_text_strict_missing_key():
    text = "alpha={RTVAR:alpha}, beta={RTVAR:beta}"
    variables = {"alpha": 1.5}
    with pytest.raises(KeyError):
        format_text(text, variables, strict=True)


def test_format_text_non_strict_leaves_placeholder():
    text = "alpha={RTVAR:alpha}, beta={RTVAR:beta}"
    variables = {"alpha": 1.5}
    assert format_text(text, variables, strict=False) == "alpha=1.5, beta={RTVAR:beta}"


def test_format_file_in_place(tmp_path: Path):
    file_path = tmp_path / "input.txt"
    file_path.write_text("value={RTVAR:value}", encoding="utf-8")
    format_file(str(file_path), {"value": 10})
    assert file_path.read_text(encoding="utf-8") == "value=10"


def test_format_files_to_output_dir(tmp_path: Path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    paths = []
    for name in ["a.txt", "b.txt"]:
        path = input_dir / name
        path.write_text("x={RTVAR:x}", encoding="utf-8")
        paths.append(str(path))

    formatted = format_files(paths, {"x": 7}, output_dir=str(output_dir))

    assert all(path.parent == output_dir for path in formatted)
    assert (output_dir / "a.txt").read_text(encoding="utf-8") == "x=7"
    assert (output_dir / "b.txt").read_text(encoding="utf-8") == "x=7"


def test_parse_params_file(tmp_path: Path):
    params = tmp_path / "params.in"
    params.write_text(
        "# comment\nalpha=1.5\nbeta = two\n\n# trailing\n",
        encoding="utf-8"
    )
    variables = _parse_params_file(str(params))
    assert variables == {"alpha": "1.5", "beta": "two"}
