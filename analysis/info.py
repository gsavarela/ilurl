"""Analyses aggregate experiement files e.g info"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-13'
import json

import numpy as np
import matplotlib.pyplot as plt

# TODO: make optional input params / discover
EMISSION_PATH = 'data/emissions/6030/'
FILENAME = 'intersection_20200217-1325031581945903.078002.9000.w.info.json'
CYCLE = 90
TIME = 9000

file = f'{EMISSION_PATH}{FILENAME}'
with open(file, 'r') as f:
    data = json.load(f)



# axes = plt.gca()


# ax1.set_ylim([-20, 0])
num_iter = len(data['per_step_returns'])
# number of cycles per iteration
N = int((TIME * num_iter) / CYCLE)
cycle_rets = np.zeros((N,), dtype=float)
cycle_vels = np.zeros((N,),  dtype=float)
cycle_vehs = np.zeros((N,),  dtype=float)

print(data.keys())
for i in range(num_iter):

    rets = data['per_step_returns'][i]
    vels = data['per_step_velocities'][i]
    vehs = data['per_step_vehs'][i]
    for t in range(0, TIME, CYCLE):
        cycle = int(i * (TIME / CYCLE) + t / CYCLE)
        cycle_vels[cycle] = np.mean(vels[t:t + CYCLE])
        cycle_rets[cycle] = np.mean(rets[t:t + CYCLE])
        cycle_vehs[cycle] = np.mean(vehs[t:t + CYCLE])

_, ax1 = plt.subplots()
ax1.set_xlabel(f'Cycles ({CYCLE} sec)')
ax1.set_ylabel('Return', color='c')
ax1.plot(cycle_rets, 'c-')
plt.title('Return per cycle')
plt.show()

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_ylabel('Speed', color='c')
ax2.set_ylabel('Count', color='b')
ax1.plot(cycle_vels, 'c-')
ax2.plot(cycle_vehs, 'r-')
plt.show()
