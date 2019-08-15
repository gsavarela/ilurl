import unittest

from ilu.core.params import QLParams
from ilu.ql.reward import RewardCalculator


class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.rcw = RewardCalculator(QLParams())

    def test_categorize(self):
        """ Tests function that converts observation space
            into categories

        (s0, c0, s1, c1, s2, c2, s3, c3)
        sX is the mean speeds on traffic light X
        cX is the mean count on traffic light X

        """
        observation_space = (0.0, 0.0, 8.75, 5.9, 12.5, 6.0, 8.76, 1.0)
        state = (0, 0, 0, 1, 1, 2, 1, 1)
        test_space = self.rcw.categorize(
            observation_space
        )

        self.assertEqual(test_space, state)

    def test_split(self):
        """ Tests function that splits the observation space

        (s0, c0, s1, c1, s2, c2, s3, c3)
        sX is the mean speeds on traffic light X
        cX is the mean count on traffic light X

        """
        observation_space = (0.0, 0.0, 8.75, 5.9, 12.5, 6.0, 8.76, 1.0)
        test_speeds, test_counts = self.rcw.split(
            observation_space
        )
        self.assertEqual(test_speeds, (0.0, 8.75, 12.5, 8.76))
        self.assertEqual(test_counts, (0.0, 5.9, 6.0, 1.0))

    def test_weighted_average(self):
        """ Computes the weighted average from
            vehicle speeds and vehicle counts

        (s0, c0, s1, c1, s2, c2, s3, c3)
        sX is the mean speeds on traffic light X
        cX is the mean count on traffic light X

        """
        observation_space = (0, 2, 1, 1, 2, 1, 0, 2)

        self.assertEqual(self.rcw.calculate(observation_space), 0.5)

    def test_costs(self):
        """Computes the costs for having some
            vehicles on each speed tier e.g
            speeds = (1, 2, 0, 1)
            ratios = (0.25, 0.5, 0.25)
            costs = (0.75, 0.5, 0.0)
        """
        observation_space = (0.0, 0.0, 8.75, 5.9, 12.5, 6.0, 8.76, 1.0)
        calc = RewardCalculator(
            QLParams(
                rewards={
                    'type': 'costs',
                    'costs': (0.75, 0.5, 0.0)
                })
        )
        self.assertEqual(calc.calculate(observation_space), 562.5)
