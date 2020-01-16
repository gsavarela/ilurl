#!/bin/bash
 python models/baseline.py grid -t 36000 -s 1 -i 100 -r 0 -e 0 -W 1

 python models/train.py grid -t 36000 -s 1 -i 100 -r 0 -e 0 -p 1 -S 40 -L 50 -W 1

 python models/train.py grid -t 36000 -s 1 -i 100 -r 0 -e 0 -p 1 -S 36 -L 54 -W 1

 python models/train.py grid -t 36000 -s 1 -i 100 -r 0 -e 0 -p 1 -S 30 -L 60 -W 1
