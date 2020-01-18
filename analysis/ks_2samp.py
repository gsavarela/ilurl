"""Implements the Komogorov Smirnov

    Are the two return samples comming from the same distribution?

""" 
__author__ = 'Guilherme Varela'
__date__ = '2020-01-12'
import json
import os

import numpy as np
from scipy.stats import ks_2samp

EMISSION_PATH = f"{os.environ['ILURL_HOME']}/data/emissions/"
# baseline_path = f"{EMISSION_PATH}intersection_20200112-1758511578851931.3494809-info.json" 

baseline_path = f"{EMISSION_PATH}intersection_20200117-2058001579294680.783177.info.json" 

# This is the true test
# test_path = f"{EMISSION_PATH}intersection_20200112-1817011578853021.163328-info.json" 
# This is another baseline test
test_path = f"{EMISSION_PATH}intersection_20200117-2109501579295390.89992.info.json"
with open(baseline_path, 'r') as f:
    baseline = json.load(f)

with open(test_path, 'r') as f:
    agent = json.load(f)

baseline_returns = np.array(baseline['returns'])
baseline_speeds = np.array(baseline['velocities'])
baseline_per_returns = np.array([ret
                                 for returns in baseline['per_step_returns']
                                 for ret in returns])

agent_returns = np.array(agent['returns'])
agent_speeds = np.array(agent['velocities'])
agent_per_returns = np.array([ret
                              for returns in agent['per_step_returns']
                              for ret in returns])

print(ks_2samp(baseline_returns, agent_returns))
print(ks_2samp(baseline_speeds, agent_speeds))
print(ks_2samp(baseline_per_returns, agent_per_returns))

