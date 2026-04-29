import os
import pathlib

ccache_config_path = os.getenv('CCACHE_CONFIGPATH')
if not ccache_config_path:
    raise RuntimeError('The CCACHE_CONFIGPATH environment variable must be set.')

# Expand ~ to actual home directory before writing config.
ccache_config_abspath = pathlib.Path(os.path.expanduser(ccache_config_path)).resolve()
ccache_config_abspath.parent.mkdir(parents=True, exist_ok=True)

print('ccache_config_path:', ccache_config_path)
print('ccache_config_path_absolute:', ccache_config_abspath)

with open(ccache_config_abspath, 'w', encoding='utf-8') as f:
    f.writelines([
        'base_dir = /project/build\n',
        'hash_dir = false\n',
    ])
