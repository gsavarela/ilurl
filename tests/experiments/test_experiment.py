import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig

from ilurl.envs.base import (TrafficLightQLEnv, ADDITIONAL_TLS_PARAMS,
                             ADDITIONAL_ENV_PARAMS, QL_PARAMS)
from ilurl.core.params import QLParams
from ilurl.scenarios.base import BaseScenario
from ilurl.core.experiment import Experiment

class TestExperiment(unittest.TestCase):
    """Tests learning add flow only on horizontal edges"""
    def setUp(self):
        sumo_args = {
            'render': False,
            'print_warnings': False,
            'sim_step': 1,
            'restart_instance': True
        }
        sim_params = SumoParams(**sumo_args)

        additional_params = {}
        additional_params.update(ADDITIONAL_ENV_PARAMS)
        additional_params.update(ADDITIONAL_TLS_PARAMS)
        # should assign long_cycle_time to street
        additional_params['long_cycle_time'] = 75
        additional_params['short_cycle_time'] = 15

        env_params = EnvParams(evaluate=True,
                               additional_params=additional_params)

        # Force flow only on horizontal edges
        scenario = BaseScenario(
            network_id='intersection',
            horizon=3600,
            initial_config=InitialConfig(
                edges_distribution=['309265401', '-238059328']
            )
        )

        ql_params = QLParams(epsilon=0.10, alpha=0.05,
                             states=('speed', 'count'),
                             rewards={'type': 'weighted_average',
                                      'costs': None},
                             num_traffic_lights=1, c=10,
                             choice_type='ucb')

        self.env = TrafficLightQLEnv(
            env_params=env_params,
            sim_params=sim_params,
            ql_params=ql_params,
            scenario=scenario
        )

    def tearDown(self):
        # terminate traci instance
        if hasattr(self, 'env'):
            self.env.terminate()

            # free data used by class
            self.env = None

    def test_experiment(self):
        experiment = Experiment(self.env)

        info = experiment.run(1, 3600)
        print(info)
        print(self.env.dpq.Q)
        import pdb
        pdb.set_trace()
