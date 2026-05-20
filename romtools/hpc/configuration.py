import os
import argparse
import warnings

try:
    import yaml
except ImportError:
    yaml = None

SCHEMA = {
    "ssh": {
        "remote": {"cli": "-r", "type": str, "help": "The remote host to connect to."},
        "user":   {"cli": "-u", "type": str, "help": "The username to use for the connection."},
        "port":   {"cli": "-p", "type": int, "help": "The port to use for the connection."},
    },
    "workflow": {
        "remote_root": {"cli": "-R", "type": str, "help": "Directory on the remote host where campaigns are staged, absolute or relative to the home directory."},
        "collect":     {"cli": "-o", "type": str, "help": "Comma-separated list of files, directories, or glob patterns to retrieve from the remote run directory. If omitted, the entire run directory is retrieved."},
    },
    "slurm": {
        "script":         {"cli": "-s", "type": str, "help": "Path to a local SLURM batch script that will be used for the job."},
        "job_name":       {"cli": "-j", "type": str, "help": "Name of the SLURM job."},
        "num_nodes":      {"cli": "-n", "type": int, "help": "Number of nodes to request for the SLURM job."},
        "tasks_per_node": {"cli": "-t", "type": int, "help": "Number of tasks to run on each node for the SLURM job."},
        "wall_time":      {"cli": "-w", "type": str, "help": "Maximum wall time for the SLURM job (format: HH:MM:SS)."},
        "partition":      {"cli": "-q", "type": str, "help": "The partition to submit the SLURM job to (e.g., batch, short)."},
        "poll_interval":  {"cli": "-P", "type": int, "help": "Seconds between squeue polls when waiting for job completion (default: 30)."},
        "account":        {"cli": "-a", "type": str, "help": "The account WCID to charge for the SLURM job."},
    },
    "output": {
        "debug": {"cli": "-d", "type": bool, "help": "Whether to enable debug logging."},
    }
}

def _normalize_collect(value):
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


class Configuration:
    """
    Handles parsing a yaml file and any supplied command-line args.

    Precedence:
      1. CLI args (overwrite YAML)
      2. YAML file values (if provided)
      3. class defaults
    """
    def __init__(self):
        # SSH configuration
        self.remote = None
        self.user = None
        self.port = 22

        # SLURM configuration
        self.script = None
        self.job_name = "hpctools_job"
        self.num_nodes = 1
        self.tasks_per_node = 1
        self.wall_time = "00:01:00"
        self.partition = "short"
        self.account = None
        self.poll_interval = 30

        # Workflow configuration
        self.remote_root = "hpctools_campaigns"

        # Output and logging configuration
        self.debug = False

        # Optional list of files/directories/globs to retrieve from the remote run directory.
        # If None, the entire run directory is retrieved.
        self.collect = None

        # Parse YAML first, then CLI overwrites YAML
        self.__parse_yaml()
        self.__parse_args()

    def __parse_yaml(self) -> None:
        """
        Loads configuration from a YAML file if one is specified on the command line.

        Accepted ways to specify YAML:
          - --input / -i PATH

        YAML may be either:
          - a flat mapping (keys match attribute names), or
          - a nested mapping with sections: ssh, slurm, workflow
        """
        pre = argparse.ArgumentParser(add_help=False)
        pre.add_argument(
            "-i", "--input",
            dest="input",
            type=str,
            default=None,
            help="Path to a YAML configuration file."
        )
        ns, _ = pre.parse_known_args()
        config_path = ns.input

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

        section_map = {k: v.keys() for k, v in SCHEMA.items()}
        is_nested = any(k in data for k in section_map.keys())

        def apply_kv(key: str, value):
            if key == "collect":
                self.collect = _normalize_collect(value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                warnings.warn(f"Warning: Unrecognized YAML key '{key}' will be ignored.", UserWarning)

        if is_nested:
            for section, _ in section_map.items():
                sec = data.get(section, {})
                if sec is None:
                    continue
                if not isinstance(sec, dict):
                    warnings.warn(f"Warning: YAML section '{section}' should be a mapping; ignoring.", UserWarning)
                    continue
                for k, v in sec.items():
                    apply_kv(k, v)

            # Also allow extra top-level flat keys alongside sections
            for k, v in data.items():
                if k in section_map:
                    continue
                apply_kv(k, v)
        else:
            for k, v in data.items():
                apply_kv(k, v)

    def __parse_args(self) -> None:
        parser = argparse.ArgumentParser(
            description="Configure the HPC dispatcher.",
            argument_default=argparse.SUPPRESS,
        )

        # Config file (so it shows up in --help; it is parsed earlier via parse_known_args)
        parser.add_argument("-i", "--input", type=str, help="Path to a YAML configuration file.")

        for group, items in SCHEMA.items():
            new_group = parser.add_argument_group(group)
            for arg_name, arg in items.items():
                new_group.add_argument(arg["cli"], f"--{arg_name}", type=arg["type"], help=arg["help"])

        args, _ = parser.parse_known_args()
        for name, value in vars(args).items():
            if name == "input":
                continue
            if name == "collect":
                self.collect = _normalize_collect(value)
            elif hasattr(self, name):
                setattr(self, name, value)
            else:
                warnings.warn(f"Warning: Unrecognized argument '{name}' will be ignored.", UserWarning)
