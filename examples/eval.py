'''Evaluation script for smart grid scenario'''

__author__ = 'Guilherme Varela'
__date__ = '2019-09-24'

from glob import glob
import pdb
import dill

from examples.smart_grid import smart_grid_example
from ilurl.core.experiment import Experiment
from ilurl.envs.green_wave_env import TrafficLightQLGridEnv

def evaluate(envs):
    for i, env in enumerate(envs):
        # HACK: This should be handled by the proper loader
        env_eval = TrafficLightQLGridEnv(env.env_params,
                                         env.sim_params,
                                         env.ql_params,
                                         env.scenario)
        #env_eval.dpq.Q = env.dpq.Q
        exp_eval = Experiment(env_eval)
        print(f"Running evaluation {i + 1}")
        # pdb.set_trace()
        info = exp_eval.run(30, 1500, show_plot=True)

if __name__ == '__main__':
    # TODO: provide command line arguments in order to customize
    # both path and examples
    paths = glob("*.pickle")
    if  not any(paths):
        raise Exception("Environment pickles must be saved on root")

    envs = []
    for path in paths:
        with open(path, mode='rb') as f:
            envs.append(dill.load(f))
    evaluate(envs)
