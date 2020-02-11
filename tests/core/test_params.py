import unittest
import numpy as np
from ilurl.core.params import QLParams


class TestQLParamsConstraints(unittest.TestCase):
    '''Tests  ilurl.core.params.QLParams '''

    # constraints testing
    def test_alpha_eq0(self):
        with self.assertRaises(ValueError):
            QLParams(alpha=0)

    def test_alpha_eq1(self):
        with self.assertRaises(ValueError):
            QLParams(alpha=1)

    def test_epsilon_eq0(self):
        '''0 < epsilon < 1'''
        with self.assertRaises(ValueError):
            QLParams(epsilon=0)

    def test_epsilon_eq1(self):
        '''0 < epsilon < 1'''
        with self.assertRaises(ValueError):
            QLParams(epsilon=1)

    def test_reward(self):
        with self.assertRaises(ValueError):
            QLParams(rewards={'type': 'x'})
    
    def test_choice_type(self):
        with self.assertRaises(ValueError):
            QLParams(choice_type='my precious choice')


class TestQLParamsAssignments(unittest.TestCase):
    '''Tests  ilurl.core.params.QLParams

    That things are being correctly assigned
    '''

    def setUp(self):
        self.ql_params = QLParams(alpha=2e-1,
                                  epsilon=1e-2,
                                  gamma=0.75,
                                  rewards={
                                      'type': 'fix',
                                      'costs': (0, 0.5, 0.75)
                                  },
                                  states=('count', )
                                  )

    def test_alpha(self):
        self.assertEqual(self.ql_params.alpha, 2e-1)

    def test_epsilon(self):
        self.assertEqual(self.ql_params.epsilon, 1e-2)

    def test_reward_type(self):
        self.assertEqual(self.ql_params.rewards.type, 'fix')

    def test_costs(self):
        self.assertEqual(self.ql_params.rewards.costs, (0, 0.5, 0.75))

    def test_max_speed(self):
        self.assertEqual(self.ql_params.max_speed, 35)

    def test_states_rank(self):
        self.assertEqual(self.ql_params.states.rank, 2)

    def test_states_depth(self):
        self.assertEqual(self.ql_params.states.depth, 3)

    def test_actions_rank(self):
        self.assertEqual(self.ql_params.actions.rank, 1)

    def test_actions_depth(self):
        self.assertEqual(self.ql_params.actions.depth, 2)


class TestQLParamsCategorize(unittest.TestCase):
    def setUp(self):
        params = QLParams(num_traffic_lights=4)
        cats = params.category_speeds
        s0, s1, s2 = cats[0], cats[0] + 0.01, cats[1]
        catc = params.category_counts
        c0, c1, c2 = catc[0], catc[1] - 0.01, catc[1]

        self.states = params.categorize_space(
            [[[s0, c0], [s0, c1]],
             [[s0, c2], [s1, c0]],
             [[s1, c1], [s1, c2]],
             [[s2, c0], [s2, c1]]]

        )
        self.speeds, self.counts = params.split_space(self.states)

    def test_categorize_space_speeds(self):
        speeds = [phase[0] for tls in self.states for phase in tls]
        self.assertEqual(speeds, [0, 0, 0, 1, 1, 1, 2, 2])

    def test_categorize_space_counts(self):
        counts = [phase[1] for tls in self.states for phase in tls]
        self.assertEqual(counts, [0, 1, 2, 0, 1, 2, 0, 1])

    def test_split_space_speeds(self):
        speeds = [phase[0] for tls in self.states for phase in tls]
        self.assertEqual(self.speeds, speeds)
        
    def test_split_space_counts(self):
        counts = [phase[1] for tls in self.states for phase in tls]
        self.assertEqual(self.counts, counts)


if __name__ == '__main__':
    unittest.main()
