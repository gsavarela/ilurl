import unittest

from ilurl.core.params import QLParams
from ilurl.core.ql.reward import RewardCalculator


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

    def test_fix(self):
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
                    'type': 'fix',
                    'costs': (0.75, 0.5, 0.0)
                })
        )
        self.assertEqual(calc.calculate(observation_space), 562.5)

    def test_score_and_queue(self):
        """Computes the score as the negative length of the average queue
        """
        observation_space = (0.0, 0.0, 5.3, 10)
        calc = RewardCalculator(
            QLParams(
                states=('queue',),
                rewards={
                    'type': 'score',
                    'costs': None
                })
        )
        self.assertEqual(calc.calculate(observation_space), -15.3)

    def test_score_and_flow(self):
        """Computes the score as the negative length of the number
            of vehicles exiting intersection
        """
        observation_space = (1.0, 10.0, 23.0, 0.0)
        calc = RewardCalculator(
            QLParams(
                states=('flow',),
                rewards={
                    'type': 'score',
                    'costs': None
                })
        )
        self.assertEqual(calc.calculate(observation_space), 34.0)

    def test_score_and_flow_queue(self):
        """Computes the score as the negative length of the number
            of vehicles exiting intersection
        """
        observation_space = (1.0, 0.0, 10.0, 0.0, 23.0, 5.3, 0.0, 10.0)
        calc = RewardCalculator(
            QLParams(
                states=('flow', 'queue',),
                rewards={
                    'type': 'score',
                    'costs': None
                })
        )
        self.assertEqual(calc.calculate(observation_space), 34.0-15.3)

    def test_score_and_flow_queue(self):
        """Computes the score as the negative length of the number
            of vehicles exiting intersection
        """
        observation_space = (0.0, 1.0, 0.0, 10.0, 5.3, 23.0, 10.0, 0.0)
        calc = RewardCalculator(
            QLParams(
                states=('queue', 'flow',),
                rewards={
                    'type': 'score',
                    'costs': None
                })
        )
        self.assertEqual(calc.calculate(observation_space), 34.0-15.3)



if __name__ == '__main__':
    unittest.main()
