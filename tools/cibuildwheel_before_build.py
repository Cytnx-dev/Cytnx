import os
import pathlib
import platform

ccache_config_path = os.getenv('CCACHE_CONFIGPATH')
if not ccache_config_path:
    raise RuntimeError('The CCACHE_CONFIGPATH environment variable must be set.')

if platform.system() == 'Linux':
    required = [
        'CMAKE_C_COMPILER_LAUNCHER',
        'CMAKE_CXX_COMPILER_LAUNCHER',
        'CCACHE_COMPILERCHECK',
    ]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        raise RuntimeError(f"Missing required Linux ccache env vars: {', '.join(missing)}")

# Expand ~ to actual home directory before writing config.
ccache_config_abspath = pathlib.Path(os.path.expanduser(ccache_config_path)).resolve()
ccache_config_abspath.parent.mkdir(parents=True, exist_ok=True)

# cibuildwheel mounts Linux sources at /project inside manylinux containers.
# On macOS/local runs, use the repository build directory instead.
if platform.system() == 'Linux':
    ccache_base_dir = '/project/build'
else:
    ccache_base_dir = str(pathlib.Path.cwd().joinpath('build').resolve())

print('ccache_config_path:', ccache_config_path)
print('ccache_config_path_absolute:', ccache_config_abspath)
print('ccache_base_dir:', ccache_base_dir)

with open(ccache_config_abspath, 'w', encoding='utf-8') as f:
    f.writelines([
        f'base_dir = {ccache_base_dir}\n',
        'hash_dir = false\n',
    ])
