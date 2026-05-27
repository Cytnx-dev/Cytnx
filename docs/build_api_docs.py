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

    # Doxyfile declares OUTPUT_DIRECTORY = "./docs/" and HTML_OUTPUT = html,
    # both relative to the working directory of the doxygen invocation.
    cwd = doxyfile.parent
    doxygen_html_dir = cwd / "docs" / "html"
    if doxygen_html_dir.exists():
        shutil.rmtree(doxygen_html_dir)

    subprocess.run(["doxygen", str(doxyfile)], cwd=cwd, check=True)

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(doxygen_html_dir), str(output_dir))

    print(f"API documentation generated at {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
