import unittest

from experiments.smart_grid import smart_grid_example
from ilu.envs.agents import TrafficLightQLGridEnv, ADDITIONAL_TLS_PARAMS


class TestGetState(unittest.TestCase):
    """Tests the traffic light function """
    def tearDown(self):
        # terminate traci instance
        if hasattr(self, 'env'):
            self.env.terminate()

            # free data used by class
            self.env = None

    def test_incoming_edges(self):

        env_params = ADDITIONAL_TLS_PARAMS.copy()
        env_params['filter_incoming_edges'] = True

        _, self.env = smart_grid_example(additional_env_params=env_params)
 
        # center0 assert
        edges = ['right0_0', 'top0_1', 'left1_0', 'bot0_0']
        self.assertEqual(
            sorted(self.env._observation_space_filter[0]),
            sorted(edges))

        # center1 assert
        edges = ['right0_1', 'top0_2', 'left1_1', 'bot0_1']
        self.assertEqual(
            sorted(self.env._observation_space_filter[1]),
            sorted(edges))

        # center2 assert
        edges = ['right1_0', 'top1_1', 'left2_0', 'bot1_0']
        self.assertEqual(
            sorted(self.env._observation_space_filter[2]),
            sorted(edges))

        # center3 assert
        edges = ['right1_1', 'top1_2', 'left2_1', 'bot1_1']
        self.assertEqual(
            sorted(self.env._observation_space_filter[3]),
            sorted(edges))

    def test_outgoing_edges(self):

        env_params = ADDITIONAL_TLS_PARAMS.copy()
        env_params['filter_incoming_edges'] = False

        _, self.env = smart_grid_example(additional_env_params=env_params)

        # center0 assert
        edges = ['left0_0', 'bot0_1', 'right1_0', 'top0_0']
        self.assertEqual(
            sorted(self.env._observation_space_filter[0]),
            sorted(edges))

        # center1 assert
        edges = ['left0_1', 'bot0_2', 'right1_1', 'top0_1']
        self.assertEqual(
            sorted(self.env._observation_space_filter[1]),
            sorted(edges))

        # center2 assert
        edges = ['left1_0', 'bot1_1', 'right2_0', 'top1_0']
        self.assertEqual(
            sorted(self.env._observation_space_filter[2]),
            sorted(edges))

        # center3 assert
        edges = ['left1_1', 'bot1_2', 'right2_1', 'top1_1']
        self.assertEqual(
            sorted(self.env._observation_space_filter[3]),
            sorted(edges))


