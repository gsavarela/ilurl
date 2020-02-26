"""Contains an experiment class for running simulations.
   2019-06-2019
   ------------
   This file was copied from flow.core.experiment in order to
   add the following features:
   * periodically save the running data: server seems to
   be restarting every 400 steps, the rewards are being changed
   radically after each restart

   * extend outputs to costumized reward functions
   * fix bug of averaging speeds when no cars are on the simulation
   """
import warnings
import datetime
import json
import logging
import os
import tempfile
import time
from collections import defaultdict

from tqdm import tqdm

import numpy as np
from flow.core.util import emission_to_csv

# TODO: Track those anoying warning
warnings.filterwarnings('ignore')

# TODO: Generalize for any parameter
ILURL_HOME = os.environ['ILURL_HOME']

EMISSION_PATH = \
    f'{ILURL_HOME}/data/emissions/'

class Experiment:
    """
    Class for systematically running simulations in any supported simulator.

    This class acts as a runner for a scenario and environment. In order to use
    it to run an scenario and environment in the absence of a method specifying
    the actions of RL agents in the network, type the following:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> exp = Experiment(env)  # for some env and scenario
        >>> exp.run(num_runs=1, num_steps=1000)

    If you wish to specify the actions of RL agents in the network, this may be
    done as follows:

        >>> rl_actions = lambda state: 0  # replace with something appropriate
        >>> exp.run(num_runs=1, num_steps=1000, rl_actions=rl_actions)

    Finally, if you would like to like to plot and visualize your results, this
    class can generate csv files from emission files produced by sumo. These
    files will contain the speeds, positions, edges, etc... of every vehicle
    in the network at every time step.

    In order to ensure that the simulator constructs an emission file, set the
    ``emission_path`` attribute in ``SimParams`` to some path.

        >>> from flow.core.params import SimParams
        >>> sim_params = SimParams(emission_path="./data")

    Once you have included this in your environment, run your Experiment object
    as follows:

        >>> exp.run(num_runs=1, num_steps=1000, convert_to_csv=True)

    After the experiment is complete, look at the "./data" directory. There
    will be two files, one with the suffix .xml and another with the suffix
    .csv. The latter should be easily interpretable from any csv reader (e.g.
    Excel), and can be parsed using tools such as numpy and pandas.

    Attributes
    ----------
    env : flow.envs.Env
        the environment object the simulator will run
    """

    def __init__(self, env, dir_path=EMISSION_PATH, train=True, policies=None):
        """Instantiate Experiment."""
        if not train and policies is None:
            raise ValueError(
                f"In validation mode an array of policies must be provided"
            )

        self.env = env
        self.train = train
        self.dir_path = dir_path
        self.policies = policies

        logging.info(" Starting experiment {} at {}".format(
            env.network.name, str(datetime.datetime.utcnow())))

        logging.info("Initializing environment.")


    def run(
            self,
            num_runs,
            num_steps,
            rl_actions=None,
            convert_to_csv=False,
            save_interval=None
    ):
        """Run the given scenario for a set number of runs and steps per run.

        Parameters
        ----------
        num_runs : int
            number of runs the experiment should perform
        num_steps : int
            number of steps to be performs in each run of the experiment
        rl_actions : method, optional
            maps states to actions to be performed by the RL agents (if
            there are any)
        convert_to_csv : bool
            Specifies whether to convert the emission file created by sumo
            into a csv file
        save_interval: int or None
            Will save and retore every refresh_interval number of runs.
            if not supplied None will only save after termination

        Returns
        -------
        info_dict : dict
            contains returns, average speed per step
        """
        # raise an error if convert_to_csv is set to True but no emission
        # file will be generated, to avoid getting an error at the end of the
        # simulation
        if convert_to_csv and self.env.sim_params.emission_path is None:
            raise ValueError(
                'The experiment was run with convert_to_csv set '
                'to True, but no emission file will be generated. If you wish '
                'to generate an emission file, you should set the parameter '
                'emission_path in the simulation parameters (SumoParams or '
                'AimsunParams) to the path of the folder where emissions '
                'output should be generated. If you do not wish to generate '
                'emissions, set the convert_to_csv parameter to False.')

        if save_interval is not None:
            print('Warning save_interval has been disabled')

        info_dict = {}
        if rl_actions is None:

            def rl_actions(*_):
                return None
        #  duration flags where in the current phase
        #  the syncronous agent is.
        is_synch = hasattr(self.env, "duration")

        rets = []
        mean_rets = []
        ret_lists = []
        vels = []
        vehs = []
        mean_vels = []
        vels_lists = []
        mean_vehs = []
        veh_lists = []
        std_vels = []
        outflows = []
        observation_spaces = []
        actions_lists = []
        for i in range(num_runs):
            vel = np.zeros(num_steps)
            veh = np.zeros(num_steps)
            logging.info("Iter #" + str(i))
            ret = 0
            ret_list = []
            actions_list = []
            vel_list = []
            veh_list = []
            state = self.env.reset()

            for j in tqdm(range(num_steps)):
                state, reward, done, _ = self.env.step(rl_actions(state))
                speeds = self.env.k.vehicle.get_speed(
                    self.env.k.vehicle.get_ids()
                )
                vel[j] = round(np.nanmean(speeds), 2)
                veh[j] = len(speeds)

                ret += reward if not(np.isnan(reward)) else 0
                ret_list.append(round(reward, 2))

                if is_synch and self.env.duration == 0.0 and j > 0:
                    observation_space = list(self.env.get_observation_space())
                    observation_spaces.append(observation_space)
                    if hasattr(self.env, 'rl_action'):
                        actions_list.append(list(self.env.rl_action))
                if done:
                    break


                # for every 100 decisions -- save Q
                if j % 9000 == 0:
                    filename = \
                        f'{self.env.network.name}.Q.{i + 1}-{int(j / 9000)}.pickle'
                    if self.train:
                        if hasattr(self.env, 'dump') and self.dir_path:
                            self.env.dump(self.dir_path,
                                          f'{self.env.network.name}.Q.{i + 1}-{int(j / 9000)}.pickle',
                                          attr_name='Q')

                    else:
                        if i < len(self.policies):
                            self.env.Q = self.policies[i]

            ret = round(ret, 2)
            rets.append(ret)
            vels.append(vel.tolist())
            vehs.append(veh.tolist())

            mean_rets.append(round(np.nanmean(ret_list), 2))
            ret_lists.append(ret_list)
            actions_lists.append(actions_list)

            veh_list.append(vehs)
            vel_list.append(vels)
            mean_vels.append(round(np.nanmean(vel), 2))
            mean_vehs.append(np.mean(veh))
            outflows.append(self.env.k.vehicle.get_outflow_rate(int(500)))
            std_vels.append(round(np.nanstd(vel), 2))
            print(f"""
                    Round {i}\treturn: {sum(ret_list):0.2f}\tavg speed:{mean_vels[-1]}
                  """)

        info_dict["id"] = self.env.network.name
        info_dict["returns"] = rets
        info_dict["velocities"] = mean_vels
        info_dict["mean_returns"] = mean_rets
        info_dict["per_step_returns"] = ret_lists
        info_dict["outflows"] = round(np.mean(outflows).astype(float), 2)
        info_dict["mean_outflows"] = round(np.mean(outflows).astype(float), 2)

        info_dict["per_step_velocities"] = vels
        info_dict["per_step_vehs"] = vehs
        info_dict["observation_spaces"] = observation_spaces
        info_dict["rl_actions"] = actions_lists
        info_dict["vehicles"] = mean_vehs

        print("Average, std return: {}, {}".format(np.nanmean(rets),
                                                   np.nanstd(rets)))
        print("Average, std speed: {}, {}".format(np.nanmean(mean_vels),
                                                  np.nanstd(mean_vels)))
        self.env.terminate()

        print('emissions', f'{self.env.sim_params.emission_path}/{self.env.network.name}')
        if self.env.sim_params.emission_path:
            # wait a short period of time to ensure the xml file is readable
            time.sleep(0.1)

            if convert_to_csv:
                emission_filename = \
                    "{0}-emission.xml".format(self.env.network.name)

                emission_path = os.path.join(
                    self.env.sim_params.emission_pathself.dir_path, 
                    emission_filename
                )

                emission_to_csv(emission_path)

        return info_dict


