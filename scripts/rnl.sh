#!/bin/bash
# This bash script installs ilu project on proprietary cluster
# TODO: FIX following ussues
# TODO: production testing
# virtualenv for python and libsm6 for sumo

# INSTALLATION ISSUES:
# ===================
# 1)  Can't really use script as there are no sudo permission
# > Installing system dependencies for SUMO
# > scripts/setup_sumo_ubuntu1804.sh: line 4: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 5: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 6: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 7: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 8: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 9: sudo: command not found
# > scripts/setup_sumo_ubuntu1804.sh: line 10: sudo: command not found
#
# 2) Paths error
# Installing sumo binaries
# ./rnl.sh: line 24: pushd: ilu/sumo_binaries/bin: No such file or directory
#
# 3) Dependencies error
# sumo: error while loading shared libraries: libxerces-c-3.2.so: cannot open shared object file: No such file or directory

python -m venv venv
source venv/bin/activate
#pip install -r requirements.txt

# Install FLOW
# this is the lastest tested version of flow
mkdir flow
git clone https://github.com/flow-project/flow.git flow &&\
cd flow && git checkout 5b30957047b && pip install -e .

# Install SUMO
# Get distro version and select proper script
# assumption this distro is ubuntu1804
chmod +x scripts/setup_sumo_ubuntu1804.sh && scripts/setup_sumo_ubuntu1804.sh && cd ..

echo "Installing sumo binaries"

mkdir -p sumo_binaries/bin
pushd ilu/sumo_binaries/bin
 wget https://akreidieh.s3.amazonaws.com/sumo/flow-0.4.0/binaries-ubuntu1804    .tar.xz
 tar -xf binaries-ubuntu1804.tar.xz
 rm binaries-ubuntu1804.tar.xz
 chmod +x *
 popd
 export PATH="$HOME/sumo_binaries/bin:$PATH"
 export SUMO_HOME="$HOME/sumo_binaries/bin"

# Proof to the user everything is up-to-date
echo "$(which sumo)"
echo "$(sumo --version)"

# Install ilu as a package 
pip install -r requirements.txt && pip install -e .
