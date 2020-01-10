"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os

from flow.core.params import SumoParams, EnvParams 
from flow.core.params import InFlows

# from flow.envs import TestEnv
from flow.envs.loop.loop_accel import AccelEnv, ADDITIONAL_ENV_PARAMS
from flow.core.experiment import Experiment

from ilurl.scenarios.intersection import IntersectionScenario

# TODO: Generalize for any parameter
DIR = \
    '/Users/gsavarela/Work/py/ilu/ilurl/data/networks/'

HORIZON = 270

if __name__ == '__main__':
    sim_params = SumoParams(render=True, sim_step=1, emission_path=DIR)

    inflows = InFlows()
    
    # def add(self,
    #         edge,
    #         veh_type,
    #         vehs_per_hour=None,
    #         probability=None,
    #         period=None,
    #         depart_lane="first",
    #         depart_speed=0,
    #         name="flow",
    #         begin=1,
    #         end=86400,
    #         number=None,
    #         **kwargs):
    env_params = EnvParams(additional_params=ADDITIONAL_ENV_PARAMS)

    scenario = IntersectionScenario(
        name='intersection'
    )


    env = AccelEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario
    )

    exp = Experiment(env=env)

    _ = exp.run(1, 300)
