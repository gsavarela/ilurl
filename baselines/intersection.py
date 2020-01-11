"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os
import json

from flow.core.params import SumoParams, EnvParams 

from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from ilurl.core.experiment import Experiment

from ilurl.scenarios.intersection import IntersectionScenario

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']
DIR = \
    f'{ILURL_HOME}/data/networks/'

NUM_ITERATIONS = 100
HORIZON = 3600
SIM_STEP = 0.1
if __name__ == '__main__':
    sim_params = SumoParams(render=False,
                            print_warnings=False,
                            sim_step=SIM_STEP,
                            restart_instance=True,
                            emission_path=DIR)

    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    scenario = IntersectionScenario(
        name='intersection',
        horizon=HORIZON,
    )


    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario
    )

    exp = Experiment(env=env)

    import time
    start = time.time()
    info_dict = exp.run(NUM_ITERATIONS, int(HORIZON / SIM_STEP))
    print(f'Elapsed time {time.time() - start}')
    infoname = '{}.info.json'.format(env.scenario.name)
    with open(infoname, 'w') as f:
        json.dump(info_dict, f)
