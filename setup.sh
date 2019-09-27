#!/usr/bin/bash
# This bash script installs ilu project within the home folder.
# TODO: pass a taget directory
# TODO: validate on a ubuntu environment

# Install pyenv 
# curl https://pyenv.run | bash

# Must install pyenv dependencies
# sudo apt-get xxx
# create a virtual env for the project
pyenv virtualenv 3.6.8 ilurl

# Install FLOW+SUMO
# Getting the last flow dependency
git clone --depth 1 https://github.com/flow-project.flow.git $HOME/flow

# Get distro version and select proper script
# assumption this distro is ubuntu1804
# TODO: Check on an ubuntu1804 system
# chmod +x $HOME/flow/scripts/setup_sumo_ubuntu1804.sh
# $HOME/flow/scripts/setup_sumo_ubuntu1804.sh

#Proof to the user everything is up-to-date
echo "$(which sumo)"
echo "$(sumo --version)"

# Setup virtual environment
pyenv activate ilurl
# Install flow dependencies
# TODO: is it really a necessary step?

# Install flow itself
pip install -e $HOME/flow
# Install ilurl dependencies
# TODO: is it really a necessary step?

# Install ilurl itself
pip install -e .
# Run a sumo test
python -m unittest discover tests/
