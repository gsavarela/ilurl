import unittest

from flow.core.params import SumoParams, EnvParams

from ilurl.envs.base import (TrafficLightQLEnv, ADDITIONAL_TLS_PARAMS,
                             ADDITIONAL_ENV_PARAMS, QL_PARAMS)
from ilurl.core.params import QLParams
from ilurl.networks.base import Network


class TestGetState(unittest.TestCase):
    """Tests the traffic light function """
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
        additional_params['long_cycle_time'] = 45
        additional_params['short_cycle_time'] = 45

        env_params = EnvParams(evaluate=True,
                               additional_params=additional_params)

        network = Network(
            network_id='intersection',
            horizon=360,
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

    def test_incoming_edges(self):
        edges = ['309265401', '-238059328', '-238059324', '383432312']
        self.assertEqual(
            sorted(self.env.incoming_edge_ids['247123161']),
            sorted(edges))

    def test_outgoing_edges(self):
        edges = ['238059328', '-309265401', '238059324', '-383432312']
        self.assertEqual(
            sorted(self.env.outgoing_edge_ids['247123161']),
            sorted(edges))


if __name__ == '__main__':
    unittest.main()
