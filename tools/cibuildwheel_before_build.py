import os
import pathlib
import platform

if platform.system() == 'Linux':
    required = [
        'CMAKE_C_COMPILER_LAUNCHER',
        'CMAKE_CXX_COMPILER_LAUNCHER',
        'CCACHE_COMPILERCHECK',
        'CCACHE_BASEDIR',
        'CCACHE_NOHASHDIR',
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required Linux ccache env vars: {', '.join(missing)}")

print('current_working_directory:', pathlib.Path.cwd())
print('build_dir_guess:', pathlib.Path.cwd().joinpath('build').resolve())
print('CCACHE_BASEDIR:', os.getenv('CCACHE_BASEDIR', ''))
print('CCACHE_NOHASHDIR:', os.getenv('CCACHE_NOHASHDIR', ''))
