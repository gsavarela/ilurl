#!/bin/bash
OUTPUT_DIRECTORY=$(python scripts/run_train.py)
echo $OUTPUT_DIRECTORY
python models/rollouts.py $OUTPUT_DIRECTORY -c 100 -s 500 -l 10000 -r 1 -p 4
python analysis/rollouts.py $(ls -d $OUTPUT_DIRECTORY/*eval.info.json | head -n 1)
