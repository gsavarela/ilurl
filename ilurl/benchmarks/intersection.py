"""Provides baseline for scenarios"""
__author__ = 'Guilherme Varela'
__date__ = '2020-01-08'

import os

from flow.core.params import SumoParams, VehicleParams
from flow.core.params import NetParams, InitialConfig, EnvParams
from flow.envs import TestEnv
from flow.scenarios import Scenario
from flow.core.experiment import Experiment

# TODO: Generalize for any parameter
DIR = \
    '/Users/gsavarela/Work/py/ilu/ilurl/data/networks/'


class TemplateScenario(Scenario):

    def specify_routes(self, net_params):
        return {
             "309265401":  ["309265401", "238059328"],
        }

if __name__ == '__main__':
    sim_params = SumoParams(render=True, sim_step=1, emission_path=DIR)
    net_params = NetParams(
        template={
            'net': os.path.join(DIR, 'intersection/intersection.net.xml'),
        },
    )

    veh_params = VehicleParams()
    veh_params.add('human', num_vehicles=1)
    env_params = EnvParams()

    initial_config = InitialConfig(
        edges_distribution=["309265401"]
    )


    scenario = TemplateScenario(
        name='Template',
        initial_config=initial_config,
        net_params=net_params,
        vehicles=veh_params
    )


    env = TestEnv(
        env_params=env_params,
        sim_params=sim_params,
        scenario=scenario
    )

    exp = Experiment(env=env)

    _ = exp.run(1, 300)
