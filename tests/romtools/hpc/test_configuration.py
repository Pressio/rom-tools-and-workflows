import sys

import pytest

from romtools.hpc.configuration import (
    SCHEMA,
    Configuration,
    ConfigurationError,
    _normalize_file_patterns,
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
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-remote", "myhost", "--hpc-user", "alice", "--hpc-port", "2222"])

    config = Configuration()

    assert config.remote == "myhost"
    assert config.user == "alice"
    assert config.port == 2222


def test_every_schema_argument_has_a_long_option(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-job-name", "longjob", "--hpc-partition", "batch"])

    config = Configuration()

    assert config.job_name == "longjob"
    assert config.partition == "batch"


def test_host_process_valueless_short_flag_does_not_raise(monkeypatch):
    """
    Regression test: "pytest -s tests/foo.py" reached the schema's "-s"
    (--script) and aborted construction with a ConfigurationError.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "-s", "-r", "-u"])

    assert Configuration().script is None


def test_defaults_reads_neither_yaml_nor_the_command_line(monkeypatch):
    """The dispatcher nobody asked for takes the schema defaults."""
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-remote", "host-value", "-c", "/nonexistent.yaml"])

    config = Configuration.defaults()

    assert config.remote is None
    assert config.job_name == "hpctools_job"


def test_config_path_argument_is_used_instead_of_the_command_line(tmp_path, monkeypatch):
    """A program that owns -c hands the dispatcher its YAML in code."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: from-argument\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-c", "mycase.deck"])

    assert Configuration(config_path=str(yaml_path)).remote == "from-argument"


def test_long_options_are_not_abbreviated(monkeypatch):
    """A host process's "--hpc-part" must not be read as "--hpc-partition"."""
    monkeypatch.setattr(sys, "argv", ["prog", "--part", "host-value"])

    assert Configuration().partition == "short"


def test_bad_value_for_a_schema_argument_raises(monkeypatch):
    """
    Regression test: a value that fails type conversion used to warn and drop
    the whole command line, so "--hpc-remote myhost" was silently lost with it.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-remote", "myhost", "--hpc-port", "not-a-port"])

    with pytest.raises(ConfigurationError, match="--hpc-port"):
        Configuration()


def test_missing_value_for_a_schema_argument_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-wall-time"])

    with pytest.raises(ConfigurationError, match="--hpc-wall-time"):
        Configuration()


@pytest.mark.parametrize("name", [n for section in SCHEMA.values() for n in section])
def test_a_bare_schema_name_is_left_to_the_host_program(monkeypatch, name):
    """
    Regression test: every schema argument was a bare long option, so a
    workflow's own --timeout, --debug or --script reconfigured the dispatcher.
    """
    monkeypatch.setattr(sys, "argv", ["prog", f"--{name}", "host-value"])

    assert getattr(Configuration(), name) == getattr(Configuration.defaults(), name)


def test_misspelled_dispatcher_option_raises_with_a_suggestion(monkeypatch):
    """
    Regression test: a switch in this schema's namespace that the parser did
    not recognize was swallowed as an extra, so the run proceeded unconfigured.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-remot", "myhost"])

    with pytest.raises(ConfigurationError, match="--hpc-remote"):
        Configuration()


def test_retired_short_flags_are_left_to_the_host_program(monkeypatch):
    """
    Regression test: "-i cfg -r host -u alice -o '*.log'" was silently dropped,
    leaving collect unset and results never retrieved. They are the host's now.
    """
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "-i", "cfg.yaml", "-r", "host", "-u", "alice", "-o", "*.log"],
    )

    config = Configuration()

    assert config.remote is None
    assert config.user is None
    assert config.collect is None


def test_missing_value_for_config_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-c"])

    with pytest.raises(ConfigurationError, match="config"):
        Configuration()


def test_config_file_is_the_only_short_switch_claimed(monkeypatch):
    """
    A host program's short switches stay its own. Only -c is claimed, and even
    -i, which it used to be, is now left to the surrounding command line.
    """
    monkeypatch.setattr(sys, "argv", ["prog", "-i", "/nonexistent/my_deck.yaml", "-o", "out"])

    assert Configuration().remote is None


def test_host_process_arguments_do_not_raise(monkeypatch):
    """Arguments the schema does not own are extras, not errors."""
    monkeypatch.setattr(sys, "argv", ["prog", "--host-only", "3", "-n", "8", "positional"])

    assert Configuration().num_nodes == 1


def test_remote_python_args_parse_from_cli(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["prog", "--hpc-python-setup", "module load python", "--hpc-python-command", "srun python3"],
    )

    config = Configuration()

    assert config.python_setup == "module load python"
    assert config.python_command == "srun python3"


def test_remote_python_defaults(monkeypatch):
    """
    Both are unset by default, so a local call() can tell "use this interpreter"
    apart from an explicit request for python3.
    """
    monkeypatch.setattr(sys, "argv", ["prog"])

    config = Configuration()

    assert config.python_setup is None
    assert config.python_command is None


def test_debug_flag_is_a_store_true_switch(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-debug"])
    assert Configuration().debug is True

    monkeypatch.setattr(sys, "argv", ["prog"])
    assert Configuration().debug is False


def test_collect_normalized_from_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-collect", "a.txt,b.log"])

    config = Configuration()

    assert config.collect == ["a.txt", "b.log"]


@pytest.mark.parametrize("body", [
    "collect: a.txt,b.log\nupload: in.dat\n",
    "workflow:\n  collect: a.txt,b.log\n  upload: in.dat\n",
])
def test_collect_and_upload_normalized_from_yaml(tmp_path, monkeypatch, body):
    """Regression test: the YAML path normalized these into an undefined name."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(body)
    monkeypatch.setattr(sys, "argv", ["prog", "-c", str(yaml_path)])

    config = Configuration()

    assert config.collect == ["a.txt", "b.log"]
    assert config.upload == ["in.dat"]


def test_yaml_flat_mapping(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: yamlhost\nuser: yamluser\njob_name: yamljob\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-c", str(yaml_path)])

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
    monkeypatch.setattr(sys, "argv", ["prog", "-c", str(yaml_path)])

    config = Configuration()

    assert config.remote == "yamlhost"
    assert config.user == "yamluser"
    assert config.job_name == "yamljob"
    assert config.user_defined == {"custom_field": 42}


def test_cli_overrides_yaml(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: yamlhost\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-c", str(yaml_path), "--hpc-remote", "clihost"])

    config = Configuration()

    assert config.remote == "clihost"


def test_unrecognized_yaml_key_warns_but_does_not_raise(tmp_path, monkeypatch):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("not_a_real_field: 5\n")
    monkeypatch.setattr(sys, "argv", ["prog", "-c", str(yaml_path)])

    with pytest.warns(UserWarning, match="not_a_real_field"):
        Configuration()


def test_missing_yaml_file_raises(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog", "-c", "/nonexistent/config.yaml"])

    with pytest.raises(FileNotFoundError):
        Configuration()


def test_config_switch_taken_from_the_host_command_line_says_so(monkeypatch):
    """
    Regression test: a program with a '-c' of its own (say a core count) had its
    value read as a YAML path, and the error explained neither where the path
    came from nor how to stop the dispatcher reading that command line.
    """
    monkeypatch.setattr(sys, "argv", ["driver.py", "-c", "4"])

    with pytest.raises(FileNotFoundError, match="--hpc-config"):
        Configuration()


def test_a_non_mapping_config_file_says_where_the_path_came_from(tmp_path, monkeypatch):
    """
    Regression test: a program whose own -c named a real file that was not a
    dispatcher config failed with "must be a mapping" and no hint that the
    dispatcher had read the switch off its command line.
    """
    deck = tmp_path / "mycase.deck"
    deck.write_text("just some text\n")
    monkeypatch.setattr(sys, "argv", ["driver.py", "-c", str(deck)])

    with pytest.raises(ValueError, match="--hpc-config"):
        Configuration()


def test_unparseable_config_file_says_where_the_path_came_from(tmp_path, monkeypatch):
    deck = tmp_path / "mycase.deck"
    deck.write_text("key: [unclosed\n")
    monkeypatch.setattr(sys, "argv", ["driver.py", "-c", str(deck)])

    with pytest.raises(ValueError, match="--hpc-config"):
        Configuration()


def test_config_is_also_spelled_with_the_prefix(tmp_path, monkeypatch):
    """--hpc-config is the collision-free spelling of -c."""
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("remote: yamlhost\n")
    monkeypatch.setattr(sys, "argv", ["prog", "--hpc-config", str(yaml_path)])

    assert Configuration().remote == "yamlhost"


def test_to_dict_returns_independent_copy(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["prog"])
    config = Configuration()

    as_dict = config.to_dict()
    as_dict["job_name"] = "mutated"

    assert config.job_name == "hpctools_job"
    assert not [k for k in as_dict if k.startswith("_")]  # no private state


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
def test_normalize_file_patterns_valid_inputs(value, expected):
    assert _normalize_file_patterns(value) == expected


def test_normalize_file_patterns_rejects_non_string_list_entries():
    with pytest.raises(ValueError):
        _normalize_file_patterns([1, 2])


def test_normalize_file_patterns_rejects_unsupported_type():
    with pytest.raises(ValueError):
        _normalize_file_patterns(123)
