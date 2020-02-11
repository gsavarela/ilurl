import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig

from ilurl.envs.base import (TrafficLightQLEnv, ADDITIONAL_TLS_PARAMS,
                             ADDITIONAL_ENV_PARAMS, QL_PARAMS)
from ilurl.core.params import QLParams
from ilurl.networks.base import Network
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
        network = Network(
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
            network=network
        )

    def tearDown(self):
        # terminate traci instance
        if hasattr(self, 'env'):
            self.env.terminate()

            # free data used by class
            self.env = None

    # @unittest.skip("Generate a favorable demand")
    def test_experiment(self):
        experiment = Experiment(self.env, dir_path=None)

        _  = experiment.run(1, 3600)
        # should opt always for action 1
        Q = self.env.dpq.Q
        print(Q)
        for s0 in (0, 1, 2):
            for s1 in (0, 1, 2):
                for c0 in (0, 1, 2):
                    for c1 in (0, 1, 2):
                        S = (s0, c0, s1, c1)
                        a0, a1 = Q[S].values()
                        # one action one at least for the state
                        assert (a0 > a1) or a0 == 0


if __name__ == '__main__':
    unittest.main()
