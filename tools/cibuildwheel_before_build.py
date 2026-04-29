import os
import pathlib
import platform

if platform.system() == 'Linux':
    required = [
        'CMAKE_C_COMPILER_LAUNCHER',
        'CMAKE_CXX_COMPILER_LAUNCHER',
        'CCACHE_COMPILERCHECK',
        'CCACHE_BASEDIR',
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"warning: ccache-related env vars are not set: {', '.join(missing)}")
        print('warning: continuing build without enforcing ccache launchers for local flexibility.')

print('current_working_directory:', pathlib.Path.cwd())
print('build_dir_guess:', pathlib.Path.cwd().joinpath('build').resolve())
print('CCACHE_BASEDIR:', os.getenv('CCACHE_BASEDIR', ''))
