#!/usr/bin/bash
# This bash script installs ilu project within the home folder.
# TODO: pass a taget directory
# TODO: validate on a ubuntu environment

# Install pyenv dependencies 
sudo apt-get update && apt-get install -y \
	make build-essential libssl-dev zlib1g-dev libbz2-dev \
	libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \ 
	libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev \
	python-openssl git && curl https://pyenv.run | bash

# Set environment variables
echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> $HOME/.bashrc &&\
echo 'eval "$(pyenv init -)"' >> $HOME/.bashrc &&\
echo 'eval "$(pyenv virtualenv-init -)"' >> $HOME/.bashrc
# echo 'export PYENV_ROOT="$HOME/.pyenv"'
# echo 'export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"'
	
# create a virtual env for the project
pyenv install 3.6.8
pyenv virtualenv 3.6.8 ilurl

# Install FLOW
# Make a new directory for flow
cd $HOME
mkdir flow && cd flow
# this is the last tested version of flow
git init && git remote add origin https://github.com/flow-project.flow.git &&\
	 && git fetch origin 5b30957047b808128c4a6091099bebc73ff78757 --depth 1 
# Install flow package
pip install -e .
# Get distro version and select proper script
# assumption this distro is ubuntu1804
# TODO: Check on an ubuntu18.04 system
chmod +x $HOME/flow/scripts/setup_sumo_ubuntu1804.sh
$HOME/flow/scripts/setup_sumo_ubuntu1804.sh

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
