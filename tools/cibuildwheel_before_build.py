import os
import pathlib
import platform

print('platform_system:', platform.system())
print('current_working_directory:', pathlib.Path.cwd())
print('build_dir_guess:', pathlib.Path.cwd().joinpath('build').resolve())
print('CMAKE_C_COMPILER_LAUNCHER:', os.getenv('CMAKE_C_COMPILER_LAUNCHER', ''))
print('CMAKE_CXX_COMPILER_LAUNCHER:', os.getenv('CMAKE_CXX_COMPILER_LAUNCHER', ''))
print('CCACHE_COMPILERCHECK:', os.getenv('CCACHE_COMPILERCHECK', ''))
print('CCACHE_BASEDIR:', os.getenv('CCACHE_BASEDIR', ''))
print('CCACHE_DIR:', os.getenv('CCACHE_DIR', ''))
