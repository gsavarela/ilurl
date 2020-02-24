"""This script makes a scatter plot from observation spaces

    Use this script to determine a discretization scheme

    USAGE:
    -----
    From root directory with files saved on root
    > python analysis/scatter.py

"""
__author__ = 'Guilherme Varela'
__date__ = '2020-02-22'
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
EMISSION_DIR = f"{ROOT_DIR}/data/experiments/0x02/"
# CONFIG_DIR = ('4545', '5040', '5434', '6030')
CONFIG_DIR = ('6030',)

if __name__ == '__main__':
    # this loop acumulates experiments
    ext = '.450000.l.info.json'
    phases = defaultdict(list)
    labels = []
    desired_velocity = None
    category_speeds = None
    category_counts = None
    for config_dir in CONFIG_DIR:
        lookup_jsons = f'{EMISSION_DIR}{config_dir}/*{ext}'
        for jf in glob(lookup_jsons):
            # Retrieves output data
            with open(jf, 'r') as f:
                output = json.load(f)

            filename = jf.replace(ext, '')
            # Retrieves agent data
            env = TrafficLightQLEnv.load(f"{filename}.pickle")

            if not labels:
                labels = env.ql_params.states_labels

            if not desired_velocity:
                additional_params = env.env_params.additional_params
                if 'target_velocity' in additional_params:
                    desired_velocity = \
                        additional_params['target_velocity']

            if not category_speeds:
                category_speeds = env.ql_params.category_speeds

            if not category_counts:
                category_counts = env.ql_params.category_counts

            # observation spaces
            observation_spaces_per_cycle = output['observation_spaces']
            for intersection_space in observation_spaces_per_cycle:
                for phase_space in intersection_space:
                    for i, phase in enumerate(phase_space):
                        phases[i] += [phase]

        _, ax = plt.subplots()
        for i, label in enumerate(labels):
            if i == 0:
                ax.set_xlabel(label)
            elif i == 1:
                ax.set_ylabel(label)


        ax.axvline(x=desired_velocity,
                   markerfacecoloralt='tab:purple',
                   label='target velocity')

        ax.vlines(category_speeds, 0, 1,
                  transform=ax.get_xaxis_transform(),
                  colors='tab:gray')

        ax.hlines(category_counts, 0, 1,
                  transform=ax.get_yaxis_transform(),
                  colors='tab:gray',
                  label='states')

        colors = ['tab:blue', 'tab:red']
        N = 0
        for i, phase in phases.items():
            x, y = zip(*phase)
            N += len(x)
            ax.scatter(x, y, c=colors[i], label=f'phase#{i}')

        split = f'{config_dir[:2]}x{config_dir[2:]}'
        ax.legend()
        ax.grid(True)
        plt.title(f'{split}: observation space (N={N})')
        plt.show()
    # Fixing random state for reproducibility
    # np.random.seed(19680801)


    # N = 100
    # r0 = 0.6
    # x = 0.9 * np.random.rand(N)
    # y = 0.9 * np.random.rand(N)
    # area = (20 * np.random.rand(N))**2  # 0 to 10 point radii
    # c = np.sqrt(area)
    # r = np.sqrt(x ** 2 + y ** 2)
    # area1 = np.ma.masked_where(r < r0, area)
    # area2 = np.ma.masked_where(r >= r0, area)

    # plt.scatter(x, y, s=area1, marker='^', c=c)
    # plt.scatter(x, y, s=area2, marker='o', c=c)
    # _, ax = plt.subplots()
    # colors = ['tab:blue', 'tab:red']
    # for i, phase in phases.items():
    #     x, y = zip(*phase)
    #     ax.scatter(x, y, c=colors[i], label=f'phase#{i}')

    # ax.legend()
    # ax.grid(True)
    # plt.show()
    # Show the boundary between the regions:
    # theta = np.arange(0, np.pi / 2, 0.01)
    # plt.plot(r0 * np.cos(theta), r0 * np.sin(theta))

    # plt.show()
    #     # plot building
    #     num_bins = 50
    #     # percentile separators: low, medium and high
    #     percentile_separators = (0.0, 25.0, 75.0, 100.0)
    #     perceptile_colors = ('yellow', 'green')
    #     for label, values in states.items():
    #         plt.figure()

    #         # mean and standard deviation of the distribution
    #         mu = np.mean(values)
    #         sigma = np.std(values)

    #         # the histogram of the data
    #         values_normalized = [
    #             round((v - mu) / sigma, 2) for v in values
    #         ]
    #         # Define quantiles for the histogram
    #         # ignore lower and higher values
    #         quantiles = np.percentile(values_normalized, percentile_separators)
    #         for i, q in enumerate(quantiles[1:-1]):
    #             color = perceptile_colors[i]
    #             p = percentile_separators[i]
    #             legend = f'p {int(p)} %'
    #             plt.axvline(x=float(q),
    #                         markerfacecoloralt=color,
    #                         label=legend)

    #         n, bins, patches = plt.hist(
    #             values_normalized,
    #             num_bins,
    #             density=mu,
    #             facecolor='blue',
    #             alpha=0.5
    #         )

    #         # add a 'best fit' line
    #         y = norm.pdf(bins, mu, sigma)
    #         plt.plot(bins, y, 'r--')
    #         plt.xlabel(label)
    #         plt.ylabel('Probability')
    #         title = f"Histogram of {label}"
    #         title = f"{title}\n$\mu$={round(mu, 2)},"
    #         title = f"{title}$\sigma$={round(sigma,2)}"
    #         plt.title(title)


    #         # Tweak spacing to prevent clipping of ylabel
    #         plt.subplots_adjust(left=0.15)
    #         print(f"#########{label}##########")
    #         print(f"min:\t{np.round(quantiles[0] * sigma + mu, 2)}")
    #         print(f"{percentile_separators[1]}\%\t{np.round(quantiles[1] * sigma + mu, 2)}")
    #         print(f"{percentile_separators[2]}\%\t{np.round(quantiles[2] * sigma + mu, 2)}")
    #         print(f"max:\t{np.round(quantiles[-1] * sigma + mu, 2)}")
    #     plt.show()
