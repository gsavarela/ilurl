"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os

from flow.core.params import SumoParams, VehicleParams, EnvParams
# from flow.envs import TestEnv
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment

from ilurl.scenarios.intersection import IntersectionScenario
# TODO: Generalize for any parameter
DIR = \
    '/Users/gsavarela/Work/py/ilu/ilurl/data/networks/'

if __name__ == '__main__':
    sim_params = SumoParams(render=True, sim_step=1, emission_path=DIR)

    veh_params = VehicleParams()
    veh_params.add('human', num_vehicles=3)
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    scenario = IntersectionScenario(
        name='intersection',
        vehicles=veh_params
    )


    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario
    )

    exp = Experiment(env=env)

    _ = exp.run(1, 300)
