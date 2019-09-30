#!/bin/bash
# This bash script installs ilu project within the home folder.
# TODO: pass a taget directory
# TODO: include other environments


sudo apt-get install virtualenv
mkdir $HOME/venvs/
virtualenv -p python3 $HOME/venvs/ilurl
source $HOME/venvs/ilurl/bin/activate



# Install FLOW
# this is the last tested version of flow
git clone https://github.com/flow-project/flow.git && cd $HOME/flow &&\
git checkout 5b30957047b &&\
pip install -e .

# Install flow package
# Get distro version and select proper script
# assumption this distro is ubuntu1804
chmod +x scripts/setup_sumo_ubuntu1804.sh
./scripts/setup_sumo_ubuntu1804.sh

# Proof to the user everything is up-to-date
source $HOME/.bashrc
echo "$(which sumo)"
echo "$(sumo --version)"

# Install ilu as a package 
cd $HOME/ilu && pip install -r requirements.txt && pip install -e .
# Run a sumo test
python -m unittest discover tests/
