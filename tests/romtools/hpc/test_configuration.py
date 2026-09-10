import re
import sys

import pytest

from romtools.hpc.configuration import (
    SCHEMA,
    Configuration,
    ConfigurationError,
    _normalize_collect,
)


def test_defaults_with_no_args_or_yaml(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])

    config = Configuration()

    assert config.remote is None
    assert config.user is None
    assert config.port == 22
    assert config.job_name == "hpctools_job"
    assert config.poll_interval == 30
    assert config.debug is False
    assert config.collect is None
    assert config.user_defined == {}


def test_cli_args_override_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-r", "myhost", "-u", "alice", "--port", "2222"])

    config = Configuration()

    assert config.remote == "myhost"
    assert config.user == "alice"
    assert config.port == 2222


def test_every_schema_argument_has_a_long_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--job_name", "longjob", "--partition", "batch"])

    config = Configuration()

    assert config.job_name == "longjob"
    assert config.partition == "batch"


def test_short_aliases_are_limited_to_per_job_overrides():
    """
    Short aliases are claimed from the host process's command line, so the
    schema keeps only the ones users vary between jobs. "-i" is reserved for
    --input and is registered separately.
    """
    aliases = {
        name: arg["cli"]
        for section in SCHEMA.values()
        for name, arg in section.items()
        if "cli" in arg
    }

    assert aliases == {
        "remote": "-r",
        "user": "-u",
        "script": "-s",
        "account": "-a",
    }


def test_schema_flags_are_unique_single_character_switches():
    """
    Regression test: multi-character single-dash flags such as "-pys" are
    prefix-ambiguous with "-p", so "-py x" aborted with "ambiguous option"
    while "-pyszz" silently parsed as "-p yszz".
    """
    flags = [
        arg["cli"]
        for section in SCHEMA.values()
        for arg in section.values()
        if "cli" in arg
    ]

    assert len(set(flags)) == len(flags)
    assert "-i" not in flags  # reserved for --input
    for flag in flags:
        assert re.fullmatch(r"-[A-Za-z]", flag), f"{flag} is not a single-character switch"


def test_host_process_short_flags_are_not_captured(monkeypatch):
    """
    Regression test: the dispatcher parses the host process's argv, so a
    workflow's own "-n 8" used to be read as SLURM num_nodes=8.
    """
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "-n", "8", "-p", "4444", "-o", "host.log", "-d", "-q", "quiet"],
    )

    config = Configuration()

    assert config.num_nodes == 1
    assert config.port == 22
    assert config.collect is None
    assert config.debug is False
    assert config.partition == "short"


def test_long_options_are_not_abbreviated(monkeypatch):
    """A host process's "--part" must not be read as this schema's "--partition"."""
    monkeypatch.setattr(sys, "argv", ["prog", "--part", "host-value"])

    assert Configuration().partition == "short"


def test_bad_value_for_a_schema_argument_raises(monkeypatch):
    """
    Regression test: a value that fails type conversion used to warn and drop
    the whole command line, so "-r myhost" was silently lost along with it.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "-r", "myhost", "--port", "not-a-port"])

    with pytest.raises(ConfigurationError, match="--port"):
        Configuration()


def test_missing_value_for_a_schema_argument_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--wall_time"])

    with pytest.raises(ConfigurationError, match="--wall_time"):
        Configuration()


def test_missing_value_for_input_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-i"])

    with pytest.raises(ConfigurationError, match="input"):
        Configuration()


def test_host_process_arguments_do_not_raise(monkeypatch):
    """Arguments the schema does not own are extras, not errors."""
    monkeypatch.setattr(sys, "argv", ["prog", "--host-only", "3", "-n", "8", "positional"])

    assert Configuration().num_nodes == 1


def test_remote_python_args_parse_from_cli(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--python_setup", "module load python", "--python_command", "srun python3"],
    )

    config = Configuration()

    assert config.python_setup == "module load python"
    assert config.python_command == "srun python3"


def test_remote_python_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])

    config = Configuration()

    assert config.python_setup is None
    assert config.python_command == "python3"


def test_debug_flag_is_a_store_true_switch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--debug"])
    assert Configuration().debug is True

    monkeypatch.setattr(sys, "argv", ["prog"])
    assert Configuration().debug is False


def test_collect_normalized_from_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--collect", "a.txt,b.log"])

    config = Configuration()

    assert config.collect == ["a.txt", "b.log"]


def test_yaml_flat_mapping(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: yamlhost\nuser: yamluser\njob_name: yamljob\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-i", str(yaml_path)])

    config = Configuration()

    assert config.remote == "yamlhost"
    assert config.user == "yamluser"
    assert config.job_name == "yamljob"


def test_yaml_nested_mapping_with_user_defined(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        "ssh:\n"
        "  remote: yamlhost\n"
        "  user: yamluser\n"
        "slurm:\n"
        "  job_name: yamljob\n"
        "user-defined:\n"
        "  custom_field: 42\n"
    )
    monkeypatch.setattr(sys, "argv", ["prog", "-i", str(yaml_path)])

    config = Configuration()

    assert config.remote == "yamlhost"
    assert config.user == "yamluser"
    assert config.job_name == "yamljob"
    assert config.user_defined == {"custom_field": 42}


def test_cli_overrides_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: yamlhost\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-i", str(yaml_path), "-r", "clihost"])

    config = Configuration()

    assert config.remote == "clihost"


def test_unrecognized_yaml_key_warns_but_does_not_raise(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("not_a_real_field: 5\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-i", str(yaml_path)])

    with pytest.warns(UserWarning, match="not_a_real_field"):
        Configuration()


def test_missing_yaml_file_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-i", "/nonexistent/config.yaml"])

    with pytest.raises(FileNotFoundError):
        Configuration()


def test_to_dict_returns_independent_copy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    config = Configuration()

    as_dict = config.to_dict()
    as_dict["job_name"] = "mutated"

    assert config.job_name == "hpctools_job"


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("a.txt,b.log", ["a.txt", "b.log"]),
        (["a.txt", " b.log "], ["a.txt", "b.log"]),
        ([], None),
    ],
)
def test_normalize_collect_valid_inputs(value, expected):
    assert _normalize_collect(value) == expected


def test_normalize_collect_rejects_non_string_list_entries():
    with pytest.raises(ValueError):
        _normalize_collect([1, 2])


def test_normalize_collect_rejects_unsupported_type():
    with pytest.raises(ValueError):
        _normalize_collect(123)
