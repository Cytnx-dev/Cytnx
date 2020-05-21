import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, distutils, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from distutils import core
from distutils.core import Distribution
from distutils.errors import DistutilsArgError
import setuptools.command.install


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]
        cmake_args += ['-DBUILD_PYTHON=ON','-DUSE_CUDA=OFF','-DUSE_MKL=ON']
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j4']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        lib_dir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='cytnx',
    version='0.5.2',
    maintainer='Kai-Hsin Wu, Yen-Hsin Wu, Ying-Jer Kao',
    maintainer_email="kaihsinwu@gmail.com",
    description='Project Cytnx',
    long_description="""This package provides cytnx: A Cross-section of Python & C++,Tensor network library """,
    packages=["cytnx","cytnx.cytnx_extension"],
    include_package_data=True,
    ext_modules=[CMakeExtension('cytnx.cytnx')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    license="GNU LGPL",
    platforms=['POSIX','MacOS'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
)
