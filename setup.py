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


#help(build_ext.copy_file)
#exit(1)

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
        cmake_args += ['-DBUILD_PYTHON=ON','-DUSE_CUDA=OFF','-DUSE_MKL=ON','-DUSE_HPTT=ON']
        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

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

        ""
        build_temp_dir = os.path.abspath(self.build_temp)
        #print(">>>build_temp_dir",build_temp_dir)
        
        # now, construct the Cpp:
        # 1. lib file (cpp)
        Cpplib_dir = os.path.join(extdir,"lib") 
        if not os.path.exists(Cpplib_dir):
            os.mkdir(Cpplib_dir)
        
        # copy libcytnx.a:
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'libcytnx' in fn:
                self.copy_file(os.path.join(build_temp_dir,fn),Cpplib_dir)
                print("[Relocate Cpp]>> find c++ dylib: ",fn)
                break

        # copy linkflags.tmp
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'linkflags.tmp' in fn:
                self.copy_file(os.path.join(build_temp_dir,fn),extdir)
                print("[Relocate linkflags.tmp]: ",fn)
                break

        # copy cxxflags.tmp
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'cxxflags.tmp' in fn:
                self.copy_file(os.path.join(build_temp_dir,fn),extdir)
                print("[Relocate cxxflags.tmp]: ",fn)
                break

        # copy version.tmp
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'version.tmp' in fn:
                self.copy_file(os.path.join(build_temp_dir,fn),extdir)
                print("[Relocate version.tmp]: ",fn)
                break

        # copy version.tmp
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'vinfo.tmp' in fn:
                self.copy_file(os.path.join(build_temp_dir,fn),extdir)
                print("[Relocate vinfo.tmp]: ",fn)
                break

        # copy hptt
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'hptt' == fn:
                self.copy_tree(os.path.join(build_temp_dir,fn),os.path.join(extdir,"hptt"))
                print("[Relocate hptt]: ",fn)
                break

        # copy cutt
        for fn in os.listdir(build_temp_dir):
            print(fn)
            if 'cutt' == fn:
                self.copy_tree(os.path.join(build_temp_dir,fn),os.path.join(extdir,"cutt"))
                print("[Relocate cutt]: ",fn)
                break


        # 2. header file (cpp)
        Cppinc_dir = os.path.join(extdir,"include")
        if not os.path.exists(Cppinc_dir):
            os.mkdir(Cppinc_dir)

        self.copy_tree(os.path.join(ext.sourcedir,"include"),Cppinc_dir)
        print("[Relocate Cpp]>> find c++ header:")
        #print(">>>sourcedir",ext.sourcedir)
        
        #print(">>>!!!")
        ""
    

def get_version():
    f = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"version.cmake"),'r')
    version = ['','','']
    for line in f.readlines():
        if 'set' in line:
            if 'MAJOR' in line:
                version[0] = line.strip().split(')')[0].split()[-1]           
            elif 'MINOR' in line:
                version[1] = line.strip().split(')')[0].split()[-1]           
            elif 'PATCH' in line:
                version[2] = line.strip().split(')')[0].split()[-1]           

    f.close()

    return "%s.%s.%s"%(version[0] ,version[1] ,version[2])




setup(
    name='cytnx',
    version="0.7.7",
    maintainer='Kai-Hsin Wu, Hsu Ke',
    maintainer_email="kaihsinwu@gmail.com",
    description='Project Cytnx',
    long_description="""This package provides cytnx: A Cross-section of Python & C++,Tensor network library """,
    packages=["cytnx"],
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
