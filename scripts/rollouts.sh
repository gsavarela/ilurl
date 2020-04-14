#!/bin/bash
# 0) output from run_train
# ['/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/intersection_20200413-2018231586805503.232706/', '/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/intersection_20200413-2018241586805504.230125/', '/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/intersection_20200413-2018251586805505.232028/', '/Users/gsavarela/Work/py/ilu/ilurl/data/emissions/intersection_20200413-2018261586805506.229337/']
# 1) must receive as parameter a path to the batch of experiments
# 2) consumes Q.1-XXX.pickle in order to make the layouts

# BATCH_PATH=$1
# NUM_PROCESS=$2
# echo $BATCH_PATH
# echo $NUM_PROCESS

echo $ILURL_HOME
EXPERIMENTS=$(python scripts/fake_train.py | tr -d '[],')
OUTPUT_DIRECTORY=$ILURL_HOME/data/experiments/$(date +%Y%m%d%H%M$S)
echo $OUTPUT_DIRECTORY
mkdir $OUTPUT_DIRECTORY
for EXPERIMENT in $EXPERIMENTS
do
    BASE_DIRECTORY=$(basename $EXPERIMENT)
    echo copy this directory $EXPERIMENT
    echo $(dirname $EXPERIMENT)
    echo $(basename $EXPERIMENT)
    #cp -r $EXPERIMENT $OUTPUT_DIRECTORY
done

# python models/rollouts.py BATCH_PATH -c 100 -s 500 -l 100000 -r 5 -p $NUM_PROCESS | xargs python analysis/rollouts.py 

