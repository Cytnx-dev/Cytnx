import os
import pathlib
import sys

import packaging.tags
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'

ccache_config_path = os.getenv('CCACHE_CONFIGPATH')

# Expand ~ to actual home directory
ccache_config_path = os.path.expanduser(ccache_config_path)
if not ccache_config_path:
    raise RuntimeError('The CCACHE_CONFIGPATH environment variable must be set.')

print("ccache_config_path:", ccache_config_path)
print("ccache_config_path:", pathlib.Path(ccache_config_path).absolute())


wheel_tag = str(next(packaging.tags.sys_tags()))
print("wheel_tag:", wheel_tag)

with open(ccache_config_path, 'w') as f:
    f.writelines([
        'base_dir = /project/build\n'
    ])
