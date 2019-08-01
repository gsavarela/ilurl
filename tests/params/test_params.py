import unittest

from ilu.core.params import QLParams


class TestQLParamsConstraints(unittest.TestCase):
    '''Tests  ilu.core.params.QLParams '''

    # constraints testing
    def test_alpha(self):
        with self.assertRaises(ValueError):
            QLParams(alpha=1)

    def test_epsilon(self):
        '''0 < epsilon < 1'''
        with self.assertRaises(ValueError):
            QLParams(epsilon=0)

    def test_reward_type(self):
        with self.assertRaises(ValueError):
            QLParams(reward_type='x')

    def test_cost_medium(self):
        with self.assertRaises(ValueError):
            '''0 < cost_medium < cost_low'''
            QLParams(reward_type='cost', cost_medium=1)

    def test_cost_low(self):
        with self.assertRaises(ValueError):
            '''0 < cost_medium < cost_low < 1'''
            QLParams(reward_type='cost', cost_low=1.5)


class TestQLParamsAssignments(unittest.TestCase):
    '''Tests  ilu.core.params.QLParams

    That things are being correctly assigned
    '''

    def setUp(self):
        self.ql_params = QLParams(alpha=2e-1,
                                  epsilon=1e-2,
                                  gamma=0.75,
                                  reward_type='cost',
                                  cost_medium=0.25,
                                  cost_low=0.5)

    def test_alpha(self):
        self.assertEqual(self.ql_params.alpha, 2e-1)

    def test_epsilon(self):
        self.assertEqual(self.ql_params.epsilon, 1e-2)

    def test_reward_type(self):
        self.assertEqual(self.ql_params.reward_type, 'cost')

    def test_cost_medium(self):
        self.assertEqual(self.ql_params.cost_medium, 0.25)

    def test_cost_low(self):
        self.assertEqual(self.ql_params.cost_low, 0.5)
