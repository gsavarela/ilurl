"""This script integrates a with an Open Street Map section

"""

__author__ = 'Guilherme Varela'
__date__ = '2019-12-12'
import os

from flow.scenarios import Scenario
from flow.core.params import InitialConfig, NetParams

SOURCES = ["309265401#0"]

SINKS = [
    "-309265401#2",
    "306967025#0",
    "238059324#0",
]

EDGES = ["212788159_0", "247123161_0", "247123161_1", "247123161_3",
         "247123161_14", "247123161_4", "247123161_5", "247123161_6",
         "247123161_15", "247123161_7", "247123161_8", "247123161_10",
         "247123161_16", "247123161_11", "247123161_12", "247123161_13",
         "247123161_17", "247123367_0", "247123367_1", "247123374_0",
         "247123374_1", "247123374_3", "247123374_9", "247123374_4",
         "247123374_5", "247123374_6", "247123374_7", "247123449_0",
         "247123449_2", "247123449_1", "247123464_0", "3928875116_0"]

SCENARIO_PATH = \
    'data/networks/sample_intersection/intersection.net.xml'


class SimpleIntersectionScenario(Scenario):
    """This class describes an OSM custom intersection scenario

        It must implement specify_routes & specify_edge_starts
    """

    def __init__(self,
                 name,
                 vehicles,
                 inflows,
                 traffic_lights):

        super(SimpleIntersectionScenario, self).__init__(
            name=name,
            vehicles=vehicles,
            net_params=NetParams(
                inflows=inflows,
                template=f'{os.getcwd()}/{SCENARIO_PATH}',
                additional_params={
                    'speed_limit': 35
                }
            ),
            initial_config=InitialConfig(
                edges_distribution=SOURCES
            ),
            traffic_lights=traffic_lights
        )

    def specify_routes(self, net_params):
        rts = {
            "309265401#0": [(["309265401#0", "238059328#0", "306967025#0"], 0.8),
                            (["309265401#0", "238059324#0"], 0.10),
                            (["309265401#0", "238059328#0",
                              "309265399#0", "96864982#1",
                              "392619842", "238059324#0"], 0.10)],
            "-306967025#2": ["-306967025#2", "-238059328#2", "-309265401#2"],
            # "96864982#0": ["96864982#0", "96864982#1", "392619842", "238059324#0"],
            # "-238059324#1": [(["-238059324#1", "-309265401#2"], 0.5), (["-238059324#1", "238059328#0", "306967025#0"], 0.5)],
            # "309265398#0": [(["309265398#0", "306967025#0"], 0.33)#, (["309265399#0", "96864982#1", "392619842", "-309265401#2"], 0.33), (["309265399#0", "96864982#1", "392619842", "238059324#0"], 0.34)]
        }
        return rts

    def specify_edge_starts(self):
        sts = [
            ("309265401#0", 77.4), ("238059328#0", 81.22),
            ("306967025#0", 131.99), ("-306967025#2", 131.99),
            ("-238059328#2", 81.22), ("-309265401#2", 77.4),
            ("96864982#0", 46.05), ("96864982#1", 82.63),
            ("392619842", 22.22), ("238059324#0", 418.00),
            ("-238059324#1", 418.00), ("309265398#0", 117.82),
            ("309265399#0", 104.56)
        ]

        return sts
