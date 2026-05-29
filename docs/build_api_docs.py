#!/usr/bin/env python3
"""Build the Doxygen API reference for the current checkout."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOXYFILE = REPO_ROOT / "docs.doxygen"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doxyfile",
        type=Path,
        default=DOXYFILE,
        help="Path to the Doxyfile (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "build" / "html" / "api",
        help="Destination directory for the generated API HTML "
        "(default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    doxyfile = args.doxyfile.resolve()
    if not doxyfile.is_file():
        print(f"error: Doxyfile not found at {doxyfile}", file=sys.stderr)
        return 1

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Feed the Doxyfile on stdin and append OUTPUT_DIRECTORY / HTML_OUTPUT so
    # Doxygen writes its HTML straight into `output_dir`, regardless of the
    # paths configured in the Doxyfile. Doxygen applies the last assignment of
    # a scalar tag, so these trailing lines override the file's own values and
    # the script never has to guess where the build landed. Relative INPUT /
    # EXAMPLE_PATH entries still resolve against the working directory, so the
    # invocation runs from the Doxyfile's directory (the repository root).
    config = doxyfile.read_text()
    config += (
        f"\nGENERATE_HTML = YES\n"
        f"OUTPUT_DIRECTORY = {output_dir.parent}\n"
        f"HTML_OUTPUT = {output_dir.name}\n"
    )
    subprocess.run(
        ["doxygen", "-"],
        input=config,
        text=True,
        cwd=doxyfile.parent,
        check=True,
    )

    print(f"API documentation generated at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
