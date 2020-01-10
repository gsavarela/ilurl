"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os

from flow.core.params import SumoParams, EnvParams 

from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from ilurl.core.experiment import Experiment
# from flow.core.experiment import Experiment

from ilurl.scenarios.intersection import IntersectionScenario

# TODO: Generalize for any parameter
DIR = \
    '/Users/gsavarela/Work/py/ilu/ilurl/data/networks/'

HORIZON = 360
SIM_STEP = 1
if __name__ == '__main__':
    sim_params = SumoParams(render=True,
                            print_warnings=False,
                            sim_step=SIM_STEP,
                            restart_instance=True)
                            # emission_path=DIR)

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

    _ = exp.run(5, HORIZON * SIM_STEP)
