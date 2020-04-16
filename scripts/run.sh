#!/bin/bash
python jobs/train.py | xargs python jobs/rollouts.py | xargs python analysis/rollouts.py
