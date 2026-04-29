import os
import pathlib

ccache_config_path = os.getenv('CCACHE_CONFIGPATH')
if not ccache_config_path:
    raise RuntimeError('The CCACHE_CONFIGPATH environment variable must be set.')

# Expand ~ to actual home directory.
pythonccache_config_path = os.path.expanduser(ccache_config_path)

print('ccache_config_path:', ccache_config_path)
print('ccache_config_path_absolute:', pathlib.Path(pythonccache_config_path).absolute())

with open(pythonccache_config_path, 'w') as f:
    f.writelines([
        'base_dir = /project/build\n',
    ])
