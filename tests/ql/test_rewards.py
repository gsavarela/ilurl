import unittest

from ilurl.core.params import QLParams
from ilurl.core.ql.reward import RewardCalculator


class TestRewardCalculator(unittest.TestCase):
    def setUp(self):
        self.rcw = RewardCalculator(QLParams())

    def test_weighted_average(self):
        """ Computes the weighted average from
            vehicle speeds and vehicle counts

        """
        observation_space = [[[0, 2], [1, 1]],
                             [[2, 1], [1, 2]],
                             [[2, 1], [1, 2]],
                             [[2, 1], [2, 2]]]


        reward = self.rcw.calculate(observation_space)
        self.assertEqual(reward, (1 + 5 * 2 + 2 * 2) / (4 * 2 + 4 * 1))

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
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

    @unittest.skip("Not implemented")
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
