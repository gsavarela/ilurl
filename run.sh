#!/bin/bash
$HOME/ilu/setup.sh
source $HOME/venvs/ilurl/bin/activate
python $HOME/ilu/experiments/smart_grid.py
deactivate


