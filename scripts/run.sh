#!/bin/bash
# EXPERIMENTS=$(python scripts/run_train.py | tr -d [],\')
# EXPERIMENTS=$(python scripts/fake_train.py)
# echo $EXPERIMENTS
OUTPUT_DIRECTORY=$(python scripts/run_train.py)
echo $OUTPUT_DIRECTORY
# mv $EXPERIMENTS $OUTPUT_DIRECTORY
# TODO: figure out a way to do this in one go
# for EXPERIMENT in $EXPERIMENTS
# do
#     mv data/emissions/$EXPERIMENT $OUTPUT_DIRECTORY
# done

python models/rollouts.py $OUTPUT_DIRECTORY -c 100 -s 500 -l 10000 -r 1 -p 4
python analysis/rollouts.py $(ls -d $OUTPUT_DIRECTORY/*eval.info.json | head -n 1)
