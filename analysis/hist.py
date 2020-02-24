"""This script makes a histogram from the info.json output from experiment

    Use this script  to determine the best categorical breakdowns

    USAGE:
    -----
    From root directory with files saved on root
    > python analysis/hist.py

    UPDATE:
    -------
    2019-12-11
        * update normpdf function
        * deprecate TrafficLightQLGridEnv in favor of TrafficQLEnv
    2020-02-20
        * swap filename for pattern matching uniting many files at once
"""
__author__ = 'Guilherme Varela'
__date__ = '2019-09-27'
# core packages
from collections import defaultdict
import json
import os
from glob import glob

# third-party libs
import dill
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# current project dependencies
from ilurl.envs.base import TrafficLightQLEnv

ROOT_DIR = os.environ['ILURL_HOME']
EMISSION_DIR = f"{ROOT_DIR}/data/emissions/"
CONFIG_DIR = ('4545', '5040', '5434', '6030')

if __name__ == '__main__':

    # this loop acumulates experiments
    ext = '.9000.w.info.json'
    states = defaultdict(list)

    for config_dir in CONFIG_DIR:
        lookup_jsons = f'{EMISSION_DIR}{config_dir}/*{ext}'
        for jf in glob(lookup_jsons):
            # file_path = f"{path}/{filename}.9000.w.info.json"
            # Retrieves output data
            with open(jf, 'r') as f:
                output = json.load(f)

            filename = jf.replace(ext,'')
            # Retrieves agent data
            env = TrafficLightQLEnv.load(f"{filename}.pickle")

            # observation spaces
            observation_spaces_per_cycle = output['observation_spaces']
            for observation_space in observation_spaces_per_cycle:
                for i, values in enumerate(env.ql_params.split_space(observation_space)):
                    label = env.ql_params.states_labels[i]
                    states[label] += values

    # plot building
    num_bins = 50
    # percentile separators: low, medium and high
    percentile_separators = (0.0, 25.0, 75.0, 100.0)
    perceptile_colors = ('yellow', 'green')
    for label, values in states.items():
        plt.figure()

        # mean and standard deviation of the distribution
        mu = np.mean(values)
        sigma = np.std(values)

        # the histogram of the data
        values_normalized = [
            round((v - mu) / sigma, 2) for v in values
        ]
        # Define quantiles for the histogram
        # ignore lower and higher values
        quantiles = np.percentile(values_normalized, percentile_separators)
        for i, q in enumerate(quantiles[1:-1]):
            color = perceptile_colors[i]
            p = percentile_separators[i]
            legend = f'p {int(p)} %'
            plt.axvline(x=float(q),
                        markerfacecoloralt=color,
                        label=legend)

        n, bins, patches = plt.hist(
            values_normalized,
            num_bins,
            density=mu,
            facecolor='blue',
            alpha=0.5
        )

        # add a 'best fit' line
        y = norm.pdf(bins, mu, sigma)
        plt.plot(bins, y, 'r--')
        plt.xlabel(label)
        plt.ylabel('Probability')
        title = f"Histogram of {label}"
        title = f"{title}\n$\mu$={round(mu, 2)},"
        title = f"{title}$\sigma$={round(sigma,2)}"
        plt.title(title)


        # Tweak spacing to prevent clipping of ylabel
        plt.subplots_adjust(left=0.15)
        print(f"#########{label}##########")
        print(f"min:\t{np.round(quantiles[0] * sigma + mu, 2)}")
        print(f"{percentile_separators[1]}\%\t{np.round(quantiles[1] * sigma + mu, 2)}")
        print(f"{percentile_separators[2]}\%\t{np.round(quantiles[2] * sigma + mu, 2)}")
        print(f"max:\t{np.round(quantiles[-1] * sigma + mu, 2)}")
    plt.show()
