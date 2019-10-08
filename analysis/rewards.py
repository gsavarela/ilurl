"""This script makes a histogram from the info.json output from experiment
USAGE:
-----
From root directory with files saved on root
> python analysis/rewards.py
"""
__author__ = 'Guilherme Varela'
__date__ = '2019-10-01'
# core packages
import json

# third-party libs
import dill
import numpy as np
import matplotlib.pyplot as plt

# current project dependencies
from ilu.envs.agents import TrafficLightQLGridEnv

# Retrieves output data
filename = "smart-grid_20191001-1737031569947823.294037.info.json"
with open(filename, 'r') as f:
    output = json.load(f)

# Retrieves agent data
filename = "smart-grid_20191001-1737031569947823.294037.pickle"
env = TrafficLightQLGridEnv.load(filename)

returns = output['returns']
velocities = output['velocities']

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Return', color='c')
ax2.set_ylabel('Speed', color='b')

ax1.plot(returns, 'c-')
ax2.plot(velocities, 'b-')
plt.show()

print(np.corrcoef(returns, velocities))
# states = {
#     state_label: []
#     for state_label in env.ql_params.states_labels}
# 
# for observation_space in observation_spaces_per_cycle:
#     for i, values in enumerate(env.ql_params.split_space(observation_space)):
#         label = env.ql_params.states_labels[i]
#         states[label] += values
# 
# # plot building
# num_bins = 10
# # percentile separators: low, medium and high
# percentile_separators = (0.0, 20.0, 75.0, 100.0)
# perceptile_colors = ('yellow', 'green')
# for label, values in states.items():
#     plt.figure()
# 
#     # mean and standard deviation of the distribution
#     mu = np.mean(values)
#     sigma = np.std(values)
# 
#     # the histogram of the data
#     values_normalized = [
#         round((v - mu) / sigma, 2) for v in values
#     ]
#     # Define quantiles for the histogram
#     # ignore lower and higher values
#     quantiles = np.percentile(values_normalized, percentile_separators)
#     for i, q in enumerate(quantiles[1:-1]):
#         color = perceptile_colors[i]
#         plt.axvline(x=float(q), markerfacecoloralt=color)
#     n, bins, patches = plt.hist(values_normalized, num_bins,density=mu, facecolor='blue', alpha= 0.5)
# 
#     # add a 'best fit' line
#     y = mlab.normpdf(bins, mu, sigma)
#     plt.plot(bins, y, 'r--')
#     plt.xlabel(label)
#     plt.ylabel('Probability')
#     plt.title(f"""Histogram of {label}
#               $\mu$={round(mu, 2)}, $\sigma$={round(sigma,2)}""")
#     # Tweak spacing to prevent clipping of ylabel
#     plt.subplots_adjust(left=0.15)
# plt.show()
