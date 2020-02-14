"""Analyses aggregate experiement files e.g info"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'

import json
import matplotlib.pyplot as plt

EMISSION_PATH = 'data/emissions/7020/'
FILENAME = 'intersection_20200213-2008121581624492.982502.90000.w.info.json'


file = f'{EMISSION_PATH}{FILENAME}'
with open(file, 'r') as f:
    data = json.load(f)



# axes = plt.gca()
_, ax1 = plt.subplots()

ax2 = ax1.twinx()

ax1.set_ylim([-20, 0])
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Return', color='c')
ax2.set_ylabel('Speed', color='b')

rets = data['per_step_returns'][-1]
vels = data['velocities'][-1]
ax1.plot(rets, 'c-')
ax2.plot(vels, 'b-')
plt.plot(rets)
plt.plot(vels)
plt.show()
