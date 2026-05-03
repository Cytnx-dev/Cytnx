import os
import pathlib
import platform
import subprocess
import sys

print('platform_system:', platform.system())
print('current_working_directory:', pathlib.Path.cwd())
print('build_dir_guess:', pathlib.Path.cwd().joinpath('build').resolve())
print('python_executable:', sys.executable)
print('python_version:', sys.version.replace('\n', ' '))
print('CMAKE_C_COMPILER_LAUNCHER:', os.getenv('CMAKE_C_COMPILER_LAUNCHER', ''))
print('CMAKE_CXX_COMPILER_LAUNCHER:', os.getenv('CMAKE_CXX_COMPILER_LAUNCHER', ''))
print('CCACHE_COMPILERCHECK:', os.getenv('CCACHE_COMPILERCHECK', ''))
print('CCACHE_BASEDIR:', os.getenv('CCACHE_BASEDIR', ''))
print('CCACHE_DIR:', os.getenv('CCACHE_DIR', ''))
print('CC:', os.getenv('CC', ''))
print('CXX:', os.getenv('CXX', ''))
print('CCACHE_DEBUG:', os.getenv('CCACHE_DEBUG', ''))
print('CCACHE_LOGFILE:', os.getenv('CCACHE_LOGFILE', ''))
print('environment_variables_begin')
for k in sorted(os.environ):
    print(f'{k}={os.environ[k]}')
print('environment_variables_end')

for cmd in (
    ['ccache', '--show-config'],
    ['ccache', '--print-stats'],
):
    print(f"\n$ {' '.join(cmd)}")
    try:
        print(subprocess.check_output(cmd, text=True))
    except Exception as exc:
        print(f'failed to run {cmd}: {exc}')
