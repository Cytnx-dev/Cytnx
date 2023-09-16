#!/bin/bash

if [ $TRAVIS_OS_NAME = 'osx' ]; then

    # Install some custom requirements on macOS
    # e.g. brew install pyenv-virtualenv
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh;
else
    # Install some custom requirements on Linux
    sudo apt update
    # We do this conditionally because it saves us some downloading if the
    # version is the same.
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;

fi

bash miniconda.sh -b -p $HOME/miniconda
echo $HOME
source "$HOME/miniconda/etc/profile.d/conda.sh"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
# Useful for debugging any issues with conda
conda info -a
#conda create -q -n test-environment python=3.8 anaconda-client conda-build=3.20.1;
conda create -q -n test-environment python=3.8 anaconda-client conda-build;
