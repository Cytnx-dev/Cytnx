import os
import pathlib
import subprocess
import sys

for key in sorted(os.environ):
    print(f'{key}={os.environ[key]}')

logfile = os.getenv('CCACHE_LOGFILE', '')
if logfile:
    pathlib.Path(logfile).parent.mkdir(parents=True, exist_ok=True)

proc = subprocess.run(['ccache', '--show-config'], capture_output=True, text=True)
if proc.stdout:
    print(proc.stdout, end='')
if proc.returncode != 0:
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end='')
    raise SystemExit(proc.returncode)
