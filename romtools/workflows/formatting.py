import argparse
import re
from pathlib import Path
from typing import Iterable, Mapping, Any, Optional, List, Dict


_RTVAR_PATTERN = re.compile(r"\{RTVAR:([A-Za-z0-9_]+)\}")


def format_text(text: str, variables: Mapping[str, Any], strict: bool = True) -> str:
    """
    Replace RTVAR placeholders in the input text with provided values.

    Placeholders are of the form: {RTVAR:variable_name}
    """
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        if var_name in variables:
            return str(variables[var_name])
        if strict:
            raise KeyError(f"Missing RTVAR value for '{var_name}'")
        return match.group(0)

    return _RTVAR_PATTERN.sub(_replace, text)


def format_file(path: str,
                variables: Mapping[str, Any],
                output_path: Optional[str] = None,
                strict: bool = True,
                encoding: str = "utf-8") -> Path:
    """
    Replace RTVAR placeholders in a file and write the result.

    If output_path is None, the input file is modified in-place.
    """
    src_path = Path(path)
    content = src_path.read_text(encoding=encoding)
    formatted = format_text(content, variables, strict=strict)
    dest_path = Path(output_path) if output_path is not None else src_path
    dest_path.write_text(formatted, encoding=encoding)
    return dest_path


def format_files(paths: Iterable[str],
                 variables: Mapping[str, Any],
                 output_dir: Optional[str] = None,
                 strict: bool = True,
                 encoding: str = "utf-8") -> List[Path]:
    """
    Replace RTVAR placeholders in multiple files.

    If output_dir is provided, formatted files are written there using the
    original filenames.
    """
    output_base = Path(output_dir) if output_dir is not None else None
    if output_base is not None:
        output_base.mkdir(parents=True, exist_ok=True)

    formatted_paths: List[Path] = []
    for path in paths:
        src_path = Path(path)
        dest_path = output_base / src_path.name if output_base is not None else src_path
        formatted_paths.append(
            format_file(str(src_path),
                        variables,
                        output_path=str(dest_path),
                        strict=strict,
                        encoding=encoding)
        )
    return formatted_paths


def _parse_vars(var_args: Iterable[str]) -> Dict[str, str]:
    variables: Dict[str, str] = {}
    for item in var_args:
        if "=" not in item:
            raise ValueError(f"Invalid --var '{item}', expected NAME=VALUE")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid --var '{item}', name cannot be empty")
        variables[name] = value
    return variables


def _parse_params_file(path: str, encoding: str = "utf-8") -> Dict[str, str]:
    variables: Dict[str, str] = {}
    for line_no, raw_line in enumerate(Path(path).read_text(encoding=encoding).splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid params line {line_no}: '{raw_line}' (expected NAME=VALUE)")
        name, value = line.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid params line {line_no}: '{raw_line}' (name cannot be empty)")
        variables[name] = value
    return variables


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replace {RTVAR:name} placeholders in files."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Input file(s) to process"
    )
    parser.add_argument(
        "--i",
        dest="input_file",
        help="Input file (single file mode)"
    )
    parser.add_argument(
        "--o",
        dest="output_file",
        help="Output file (single file mode)"
    )
    parser.add_argument(
        "--var",
        action="append",
        default=[],
        help="Variable assignment NAME=VALUE (can be repeated)"
    )
    parser.add_argument(
        "--params",
        help="Path to params.in file containing NAME=VALUE pairs"
    )
    parser.add_argument(
        "--parmams",
        dest="params_typo",
        help="Path to params.in file containing NAME=VALUE pairs"
    )
    parser.add_argument(
        "--output",
        help="Output file (only valid with a single input file)"
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for multiple files"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Fail on missing variables (default)"
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Leave unknown placeholders unchanged"
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="File encoding (default: utf-8)"
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    variables: Dict[str, str] = {}
    if args.input_file:
        if args.files:
            parser.error("Use either positional files or --i, not both")
        args.files = [args.input_file]
    if args.output_file:
        if args.output:
            parser.error("Use either --o or --output, not both")
        args.output = args.output_file

    params_path = args.params or args.params_typo
    if params_path is None and Path("params.in").is_file():
        params_path = "params.in"
    if params_path:
        variables.update(_parse_params_file(params_path, encoding=args.encoding))
    variables.update(_parse_vars(args.var))

    if args.output and args.output_dir:
        parser.error("Use either --output or --output-dir, not both")
    if args.output and len(args.files) != 1:
        parser.error("--output requires exactly one input file")
    if not args.files:
        parser.error("No input files provided")

    if args.output:
        format_file(
            args.files[0],
            variables,
            output_path=args.output,
            strict=args.strict,
            encoding=args.encoding,
        )
    elif args.output_dir:
        format_files(
            args.files,
            variables,
            output_dir=args.output_dir,
            strict=args.strict,
            encoding=args.encoding,
        )
    else:
        format_files(
            args.files,
            variables,
            output_dir=None,
            strict=args.strict,
            encoding=args.encoding,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
