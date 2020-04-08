#!/bin/bash
# 1) must receive as parameter a path to the batch of experiments
# 2) consumes Q.1-XXX.pickle in order to make the layouts

BATCH_PATH=$1
NUM_PROCESS=$2
python models/rollouts.py BATCH_PATH -c 100 -s 500 -l 100000 -r 5 -p $NUM_PROCESS | xargs python analysis/rollouts.py 

