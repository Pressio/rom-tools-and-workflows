import os
import argparse
import warnings

try:
    import yaml
except ImportError:
    yaml = None

# -----------------------------------------------------------------------------
# Core configuration schema (defaults specified in the Configuration class)
# -----------------------------------------------------------------------------

SCHEMA = {
    "ssh": {
        "remote": {"type": str, "help": "The remote host to connect to."},
        "user":   {"type": str, "help": "The username to use for the connection."},
        "port":   {"type": int, "help": "The port to use for the connection."},
    },
    "workflow": {
        "remote_root":    {"type": str, "help": "Directory on the remote host where campaigns are staged, absolute or relative to the home directory."},
        "collect":        {"type": str, "help": "Comma-separated list of files, directories, or glob patterns to retrieve from the remote run directory. If omitted, nothing is retrieved."},
        "upload":         {"type": str, "help": "Comma-separated list of files, directories, or glob patterns to place in the run directory before work starts, uploaded to the remote host or copied from the current directory. If omitted, nothing is uploaded."},
        "python_setup":   {"type": str, "help": "Shell commands that set up the environment before invoking Python (e.g. loading modules or activating a virtual environment)."},
        "python_command": {"type": str, "help": "Command that invokes the Python with the necessary libraries installed (remote default: python3). Setting either of these makes a local call() run in a subprocess rather than this interpreter."}
    },
    "slurm": {
        "script":         {"type": str, "help": "Path to a local SLURM batch script that will be used for the job."},
        "account":        {"type": str, "help": "The account WCID to charge for the SLURM job."},
        "job_name":       {"type": str, "help": "Name of the SLURM job."},
        "num_nodes":      {"type": int, "help": "Number of nodes to request for the SLURM job."},
        "tasks_per_node": {"type": int, "help": "Number of tasks to run on each node for the SLURM job."},
        "wall_time":      {"type": str, "help": "Maximum wall time for the SLURM job (format: HH:MM:SS)."},
        "partition":      {"type": str, "help": "The partition to submit the SLURM job to (e.g., batch, short)."},
        "poll_interval":  {"type": int, "help": "Seconds between squeue polls when waiting for job completion (default: 30)."},
        "timeout":        {"type": float, "help": "Time until giving up retrieving job's sacct exit code."},
    },
    "output": {
        "debug": {"type": bool, "help": "Whether to enable debug logging."},
    },
}

# -----------------------------------------------------------------------------
# Error handling for parser
# -----------------------------------------------------------------------------

class ConfigurationError(Exception):
    """Raised when the command line cannot be parsed against SCHEMA."""


class _RaisingParser(argparse.ArgumentParser):
    """
    An ArgumentParser that raises instead of exiting the process.

    parse_known_args() returns the surrounding program's own arguments as
    extras, so error() is reached only for an option this schema owns: a long
    option, or the single short switch -c. A program that must not have its
    command line read at all builds its Configuration with argv=[].
    """

    def error(self, message):
        raise ConfigurationError(
            f"{message}. Run 'python -m romtools.hpc' to see the "
            "dispatcher's configuration arguments."
        )

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _normalize_file_patterns(value):
    """
    Normalize collect specifications into a list of strings.

    Accepted forms:
      - None
      - "foo.txt,*.log,results/"
      - ["foo.txt", "*.log", "results/"]

    Returns:
      - None if unspecified
      - list[str] if specified
    """
    if value is None:
        return None

    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        return items or None

    if isinstance(value, list):
        items = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(
                    f"Invalid collect entry {item!r}; all entries must be strings."
                )
            item = item.strip()
            if item:
                items.append(item)
        return items or None

    raise ValueError(
        f"Invalid collect value {value!r}; expected a string or list of strings."
    )

def _add_config_file_arg(parser, default=argparse.SUPPRESS):
    """The config file, and the one short switch this schema claims."""
    parser.add_argument(
        "-c", "--config",
        dest="config",
        type=str,
        default=default,
        help="Path to a YAML configuration file.",
    )

def _add_value_param(grp, arg_name, arg):
    grp.add_argument(
        f"--{arg_name}",
        dest=arg_name,
        type=arg["type"],
        default=argparse.SUPPRESS,
        help=arg["help"],
    )

def _add_flag_param(grp, arg_name, arg):
    grp.add_argument(
        f"--{arg_name}",
        dest=arg_name,
        action="store_true",
        default=argparse.SUPPRESS,
        help=arg["help"],
    )

def _add_schema_arg(grp, item):
    name, arg = item
    if arg["type"] == bool:
        _add_flag_param(grp, name, arg)
    else:
        _add_value_param(grp, name, arg)

def _build_parser() -> _RaisingParser:
    """The parser for SCHEMA, used both to read the command line and to print help."""
    # add_help=False: building a dispatcher must not claim -h from the surrounding program
    # allow_abbrev=False: its "--part" must not be read as this schema's "--partition"
    parser = _RaisingParser(
        description="Configure the HPC dispatcher.",
        argument_default=argparse.SUPPRESS,
        allow_abbrev=False,
        add_help=False,
    )

    # Config file (parsed earlier via parse_known_args; added here so it shows up in help)
    _add_config_file_arg(parser)

    for group, items in SCHEMA.items():
        if not items:
            continue

        new_grp = parser.add_argument_group(group)
        for arg in items.items():
            _add_schema_arg(new_grp, arg)

    return parser

def print_help() -> None:
    """Print the dispatcher's configuration arguments."""
    _build_parser().print_help()

# -----------------------------------------------------------------------------
# Main class holding configuration for dispatchers
# -----------------------------------------------------------------------------

class Configuration:
    """
    Handles parsing a yaml file and any supplied command-line args.

    Precedence:
      1. CLI args (overwrite YAML)
      2. YAML file values (if provided)
      3. Class defaults

    Arguments:
        argv: Argument list to parse instead of the real process argv
            (sys.argv[1:]). Pass an explicit list (e.g. []) to build a
            Configuration without reading the host process's CLI args --
            useful for embedding a dispatcher in a program whose own
            command line is not meant to configure it.
    """
    def __init__(self, argv: list = None):
        self._argv = argv

        # SSH configuration
        self.remote = None
        self.user = None
        self.port = 22

        # Python configuration for call(); unset means the current interpreter
        self.python_setup = None
        self.python_command = None

        # SLURM configuration
        self.script = None
        self.job_name = "hpctools_job"
        self.num_nodes = 1
        self.tasks_per_node = 1
        self.wall_time = "00:01:00"
        self.partition = "short"
        self.account = None
        self.poll_interval = 30
        self.timeout = 240

        # Workflow configuration
        self.remote_root = "hpctools_campaigns"

        # Output and logging configuration
        self.debug = False

        # Optional list of files/directories/globs to retrieve from the remote run directory.
        # If None, the entire run directory is retrieved.
        self.collect = None
        self.upload = None

        # User-defined fields loaded only from YAML "user-defined"
        self.user_defined = {}

        # Parse YAML first, then CLI overwrites YAML
        self.__parse_yaml()
        self.__parse_args()

    def __parse_yaml(self) -> None:
        """
        Loads configuration from a YAML file if one is specified on the command line.

        Accepted ways to specify YAML:
          - --config / -c PATH

        YAML may be either:
          - a flat mapping (keys match attribute names), or
          - a nested mapping with sections: ssh, slurm, workflow, output, user-defined

        The special "user-defined" section must be a mapping/dictionary and is stored
        as-is in self.user_defined. Its contents are not interpreted as individual
        configuration attributes.
        """
        pre = _RaisingParser(add_help=False, allow_abbrev=False)
        _add_config_file_arg(pre, default=None)
        ns, _ = pre.parse_known_args(self._argv)
        config_path = ns.config

        if not config_path:
            return

        if yaml is None:
            raise RuntimeError(
                "PyYAML is required to load a config file. Install it with: pip install pyyaml"
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("YAML config must be a mapping/dictionary at the top level.")

        section_names = set(SCHEMA.keys()) | {"user-defined"}
        is_nested = any(k in data for k in section_names)

        if is_nested:
            for section in SCHEMA.keys():
                sec = data.get(section, {})
                if sec is None:
                    continue
                if not isinstance(sec, dict):
                    warnings.warn(
                        f"Warning: YAML section '{section}' should be a mapping; ignoring.",
                        UserWarning
                    )
                    continue
                for k, v in sec.items():
                    self._apply_setting(k, v, f"YAML key '{k}'")

            user_defined_section = data.get("user-defined", {})
            if user_defined_section is None:
                pass
            elif not isinstance(user_defined_section, dict):
                warnings.warn(
                    "Warning: YAML section 'user-defined' should be a mapping; ignoring.",
                    UserWarning,
                )
            else:
                self.user_defined.update(user_defined_section)

            # Also allow extra top-level flat keys alongside sections,
            # except for the reserved nested section "user-defined".
            for k, v in data.items():
                if k in section_names:
                    continue
                self._apply_setting(k, v, f"YAML key '{k}'")
        else:
            for k, v in data.items():
                if k == "user-defined":
                    if not isinstance(v, dict):
                        warnings.warn(
                            "Warning: YAML key 'user-defined' should be a mapping; ignoring.",
                            UserWarning,
                        )
                    else:
                        self.user_defined.update(v)
                else:
                    self._apply_setting(k, v, f"YAML key '{k}'")

    def __parse_args(self) -> None:
        args, _ = _build_parser().parse_known_args(self._argv)

        for name, value in vars(args).items():
            if name != "config":
                self._apply_setting(name, value, f"argument '{name}'")

    def _apply_setting(self, key: str, value, source: str) -> None:
        """Store one setting, normalizing the pattern lists and warning on unknown keys."""
        if key in ("collect", "upload"):
            setattr(self, key, _normalize_file_patterns(value))
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            warnings.warn(f"Warning: Unrecognized {source} will be ignored.", UserWarning)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
